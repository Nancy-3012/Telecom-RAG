import json
import random
import re
import os

from retriever import retrieve_chunks
from llm_setup import generate

METADATA_PATH = "data/processed/metadata.json"
SAMPLE_SIZE = 20  # start small to confirm the script works, raise later for a fuller report
TOP_K = 10

random.seed(42)  # reproducible sample across runs


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_entry(entry):
    """Pulls the question, option lines, and ground-truth option number
    back out of the stored chunk text."""
    lines = entry["text"].split("\n")
    question = lines[0].replace("Question:", "").strip()
    options = [l for l in lines if l.lower().startswith("option")]
    answer_line = next((l for l in lines if l.startswith("Correct Answer:")), "")
    gt_match = re.search(r"option (\d+)", answer_line)
    ground_truth = int(gt_match.group(1)) if gt_match else None
    return question, options, ground_truth


def paraphrase_question(question):
    """This is the actual fix -- rewords the question so the retrieval test
    isn't just matching the exact text it already indexed."""
    prompt = (
        "Reword the following telecom question in different words, keeping "
        "the exact same meaning. Output ONLY the reworded question, nothing else.\n\n"
        f"Original: {question}"
    )
    return generate(prompt, max_tokens=100, temperature=0.7).strip().strip('"')


def parse_chosen_option(model_output, num_options):
    match = re.search(r"\b([1-9])\b", model_output)
    if match:
        n = int(match.group(1))
        if 1 <= n <= num_options:
            return n
    return None


def run_eval(samples):
    ranks = []
    correct = 0
    scored = 0

    for i, entry in enumerate(samples):
        question, options, ground_truth = parse_entry(entry)
        if ground_truth is None:
            continue

        paraphrase = paraphrase_question(question)
        results = retrieve_chunks(paraphrase, k=TOP_K)

        # where does the originally-matching entry land in these results?
        rank = None
        for r, res in enumerate(results, start=1):
            if res["id"] == entry["id"]:
                rank = r
                break
        ranks.append(rank)

        # answer accuracy: use whatever context the paraphrased query
        # actually retrieved -- not a guaranteed hit, this is realistic
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

        is_correct = chosen == ground_truth
        correct += int(is_correct)
        scored += 1

        status = "correct" if is_correct else "wrong"
        print(f"[{i+1}/{len(samples)}] rank={rank} predicted={chosen} actual={ground_truth} ({status})")
        print(f"    paraphrase: {paraphrase[:80]}")

    mrr = sum((1 / r) if r else 0 for r in ranks) / len(ranks)
    top1 = sum(1 for r in ranks if r == 1) / len(ranks)
    top3 = sum(1 for r in ranks if r and r <= 3) / len(ranks)
    top5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)

    return {
        "mrr": round(mrr, 4),
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "top5_accuracy": round(top5, 4),
        "recall_at_5": round(top5, 4),
        "answer_accuracy": round(correct / scored, 4) if scored else 0,
        "num_samples": len(samples),
        "methodology": "Queries are LLM-generated paraphrases of the original "
                        "questions, not verbatim text -- avoids the exact-match "
                        "leakage from the earlier evaluation.",
    }


if __name__ == "__main__":
    import traceback

    try:
        metadata = load_metadata()
        samples = random.sample(metadata, min(SAMPLE_SIZE, len(metadata)))

        print(f"Running realistic evaluation on {len(samples)} questions "
              f"(paraphrased queries, not verbatim text)...\n")

        results = run_eval(samples)

        print("\n=== Final Results ===")
        for k, v in results.items():
            print(f"{k}: {v}")

        with open("evaluation_results_v2.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved to evaluation_results_v2.json")
    except Exception:
        traceback.print_exc()
    finally:
        # Always force-exit here, success or failure -- skips the buggy
        # llama.cpp Metal cleanup path on normal Python shutdown either way.
        os._exit(0)