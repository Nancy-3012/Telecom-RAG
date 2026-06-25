import json
import random
import re
import os
import traceback

from retriever import retrieve_chunks
from llm_setup import generate

METADATA_PATH = "data/processed/metadata.json"
RESULTS_PATH = "evaluation_results_v2.json"
SAMPLE_SIZE = 50    # total questions to eventually cover across multiple runs
BATCH_SIZE = 5      # how many questions THIS run processes before exiting cleanly
TOP_K = 10

random.seed(42)  # same sample set every run, so batches line up correctly


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {"completed": [], "easy": [], "hard": []}


def save_progress(progress):
    with open(RESULTS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def parse_entry(entry):
    lines = entry["text"].split("\n")
    question = lines[0].replace("Question:", "").strip()
    options = [l for l in lines if l.lower().startswith("option")]
    answer_line = next((l for l in lines if l.startswith("Correct Answer:")), "")
    gt_match = re.search(r"option (\d+)", answer_line)
    ground_truth = int(gt_match.group(1)) if gt_match else None
    return question, options, ground_truth


def paraphrase_easy(question):
    prompt = (
        "Reword the following telecom question in different words, keeping "
        "the exact same meaning. Output ONLY the reworded question, nothing else.\n\n"
        f"Original: {question}"
    )
    return generate(prompt, max_tokens=100, temperature=0.7).strip().strip('"')


def paraphrase_hard(question):
    prompt = (
        "Reword the following telecom question so it asks the same thing, "
        "but avoid reusing the exact technical acronyms, protocol names, or "
        "proper nouns from the original wherever possible -- describe the "
        "underlying concept in plain, descriptive language instead. "
        "Output ONLY the reworded question, nothing else.\n\n"
        f"Original: {question}"
    )
    return generate(prompt, max_tokens=120, temperature=0.7).strip().strip('"')


def parse_chosen_option(model_output, num_options):
    match = re.search(r"\b([1-9])\b", model_output)
    if match:
        n = int(match.group(1))
        if 1 <= n <= num_options:
            return n
    return None


def test_one_query(entry, question, options, ground_truth, query_text):
    results = retrieve_chunks(query_text, k=TOP_K)

    rank = None
    for r, res in enumerate(results, start=1):
        if res["id"] == entry["id"]:
            rank = r
            break

    context = "\n\n".join(c["text"] for c in results[:5])
    options_text = "\n".join(options)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n{options_text}\n\n"
        "Based on the context, which option is correct? "
        "Reply with ONLY the option number, nothing else."
    )
    response = generate(prompt, max_tokens=10, temperature=0.0)
    chosen = parse_chosen_option(response, len(options))

    return rank, (chosen == ground_truth)


def summarize(records):
    n = len(records)
    if n == 0:
        return {"mrr": 0, "top1_accuracy": 0, "top3_accuracy": 0, "top5_accuracy": 0,
                "recall_at_5": 0, "answer_accuracy": 0, "num_samples": 0}
    ranks = [r["rank"] for r in records]
    corrects = [r["correct"] for r in records]
    mrr = sum((1 / r) if r else 0 for r in ranks) / n
    top1 = sum(1 for r in ranks if r == 1) / n
    top3 = sum(1 for r in ranks if r and r <= 3) / n
    top5 = sum(1 for r in ranks if r and r <= 5) / n
    accuracy = sum(corrects) / n
    return {
        "mrr": round(mrr, 4), "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4), "top5_accuracy": round(top5, 4),
        "recall_at_5": round(top5, 4), "answer_accuracy": round(accuracy, 4),
        "num_samples": n,
    }


if __name__ == "__main__":
    try:
        metadata = load_metadata()
        all_samples = random.sample(metadata, min(SAMPLE_SIZE, len(metadata)))

        progress = load_progress()
        completed_ids = set(progress["completed"])
        remaining = [e for e in all_samples if e["id"] not in completed_ids]
        batch = remaining[:BATCH_SIZE]

        if not batch:
            print("All samples already done.")
        else:
            print(f"Processing {len(batch)} questions this run "
                  f"({len(completed_ids)}/{len(all_samples)} already completed)...\n")

            for entry in batch:
                question, options, ground_truth = parse_entry(entry)
                if ground_truth is None:
                    progress["completed"].append(entry["id"])
                    continue

                easy_q = paraphrase_easy(question)
                easy_rank, easy_ok = test_one_query(entry, question, options, ground_truth, easy_q)

                hard_q = paraphrase_hard(question)
                hard_rank, hard_ok = test_one_query(entry, question, options, ground_truth, hard_q)

                progress["easy"].append({"rank": easy_rank, "correct": easy_ok})
                progress["hard"].append({"rank": hard_rank, "correct": hard_ok})
                progress["completed"].append(entry["id"])

                print(f"[{len(progress['completed'])}/{len(all_samples)}] "
                      f"easy: rank={easy_rank} {'ok' if easy_ok else 'wrong'}"
                      f"  |  hard: rank={hard_rank} {'ok' if hard_ok else 'wrong'}")

                save_progress(progress)  # saved after EVERY question, not just at the end

        remaining_count = len(all_samples) - len(progress["completed"])
        print(f"\nCompleted: {len(progress['completed'])}/{len(all_samples)}"
              f"  ({remaining_count} remaining -- just run this script again to continue)\n")
        print(json.dumps({
            "easy_paraphrase": summarize(progress["easy"]),
            "hard_paraphrase": summarize(progress["hard"]),
        }, indent=2))

    except Exception:
        traceback.print_exc()
    finally:
        os._exit(0)