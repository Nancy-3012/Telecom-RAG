import json
import faiss
from sentence_transformers import SentenceTransformer

from Troubleshooting_facts import TROUBLESHOOTING_FACTS

EMBEDDING_MODEL = "BAAI/bge-large-en"   # must match embeddings.py exactly
INDEX_PATH = "data/processed/faiss_index.bin"
METADATA_PATH = "data/processed/metadata.json"


def add_facts_to_index():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loading existing FAISS index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    print(f"Index currently has {index.ntotal} vectors")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Metadata currently has {len(metadata)} entries")

    # Skip re-adding if this was already run before (checked by id prefix)
    existing_ids = {e["id"] for e in metadata}
    new_facts = [f for f in TROUBLESHOOTING_FACTS if f["id"] not in existing_ids]

    if not new_facts:
        print("\nTroubleshooting facts already added previously -- nothing to do.")
        return

    print(f"\nEmbedding {len(new_facts)} new troubleshooting facts...")
    texts = [f["text"] for f in new_facts]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    index.add(embeddings)
    metadata.extend(new_facts)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print(f"\nDone. Index now has {index.ntotal} vectors, "
          f"metadata has {len(metadata)} entries.")


if __name__ == "__main__":
    add_facts_to_index()