def chunk_text(text, max_words=400, overlap_words=50):
    """
    Splits text into overlapping word-based chunks if it's longer than
    max_words. Short texts (like a single TeleQnA question+answer) pass
    through unchanged as a single chunk -- there's nothing to cut up if
    the whole entry is already smaller than one chunk.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += max_words - overlap_words

    return chunks


def chunk_entries(entries, max_words=400, overlap_words=50):
    """
    Takes the {id, text, source} entries from ingest.py and applies
    chunk_text() to each. For entries already under max_words (true for
    essentially all of TeleQnA), this returns them untouched as a single
    chunk. Long documents -- e.g. raw 3GPP spec text, if you add it later --
    would actually get split here.
    """
    chunked = []
    split_count = 0

    for entry in entries:
        pieces = chunk_text(entry["text"], max_words, overlap_words)
        if len(pieces) > 1:
            split_count += 1
        for i, piece in enumerate(pieces):
            chunked.append({
                "id": f"{entry['id']}_chunk{i}" if len(pieces) > 1 else entry["id"],
                "text": piece,
                "source": entry["source"],
            })

    print(f"Chunked {len(entries)} entries into {len(chunked)} chunks "
          f"({split_count} entries needed splitting)")
    return chunked


if __name__ == "__main__":
    from ingest import load_teleqna
    entries = load_teleqna()
    chunked = chunk_entries(entries)

    lengths = [len(e["text"].split()) for e in entries]
    print(f"\nTeleQnA entry length stats (words): "
          f"min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")