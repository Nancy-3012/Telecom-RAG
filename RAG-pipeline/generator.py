from retriever import retrieve_chunks
from llm_setup import generate

# Carried over from the Colab notebook -- already includes anti-hallucination
# guardrails (no inventing spec numbers/titles not present in the context).
PROMPT_TEMPLATE = """You are a telecom RAN expert assistant.
Answer the question using ONLY the context provided below.
Do NOT invent, guess, or cite any document numbers, specification IDs,
release numbers, or titles that are not word-for-word present in the context below.
If you reference a source, only use the exact source label given (e.g. "TeleQnA/Standards specifications").
Do not add a "Reference:" section unless that exact reference text appears in the context.
If the context does not contain the answer, say "I don't have enough information."
Keep your answer concise -- 2 to 4 sentences.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question, k=5):
    """
    Full RAG pipeline: retrieve relevant chunks, then ask the LLM to answer
    using only that context. Returns the answer plus the sources used.
    """
    chunks = retrieve_chunks(question, k=k)

    if not chunks:
        return {"answer": "I don't have enough information.", "sources": []}

    context = "\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    answer = generate(prompt)
    sources = list({c["source"] for c in chunks})

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    test_questions = [
        "What is the handover procedure in 5G NR?",
        "What does PDCP stand for?",
        "What is the purpose of the RLC layer?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = answer_question(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")

    # See llm_setup.py for why -- skips a known llama.cpp Metal cleanup bug
    # that fires on exit, after all the real work above is already done.
    import os
    os._exit(0)