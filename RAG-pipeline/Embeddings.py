import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingest import load_teleqna

EMBEDDING_MODEL = "BAAI/bge-large-en"
OUTPUT_DIR = "data/processed"
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.json")

BATCH_SIZE = 500  # entries processed and saved per batch -- lower this if you want even more frequent saves


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"processed": 0}


def save_checkpoint(processed):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"processed": processed}, f)


def build_faiss_index():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TeleQnA entries...")
    entries = load_teleqna()
    texts = [e["text"] for e in entries]
    total = len(entries)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    checkpoint = load_checkpoint()
    start = checkpoint["processed"]

    if start > 0 and os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print(f"Resuming from entry {start}/{total} (found existing checkpoint)")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        print("Starting fresh")
        start = 0
        index = None
        metadata = []

    while start < total:
        end = min(start + BATCH_SIZE, total)
        print(f"\nEmbedding entries {start} to {end} of {total}...")

        batch_texts = texts[start:end]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        if index is None:
            dimension = batch_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

        index.add(batch_embeddings)
        metadata.extend([
            {"id": e["id"], "text": e["text"], "source": e["source"]}
            for e in entries[start:end]
        ])

        # save progress after every batch -- stopping now keeps everything done so far
        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        save_checkpoint(end)

        print(f"Saved progress: {end}/{total} entries embedded and written to disk")
        start = end

    print(f"\nDone. Final index has {index.ntotal} vectors.")


if __name__ == "__main__":
    build_faiss_index()