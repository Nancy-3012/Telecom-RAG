from retriever import retrieve_chunks
from llm_setup import generate

RCA_PROMPT_TEMPLATE = """You are a telecom RAN troubleshooting expert.
An anomaly has been detected in the network. Use the context below (from
3GPP standards and technical documentation) to explain the likely cause
and recommend next steps.
Do NOT invent specification numbers or facts not present in the context.
If the context does not clearly explain this anomaly, say so honestly
rather than guessing.

Context:
{context}

Anomaly: {anomaly_description}

Likely root cause and recommendation:"""


def explain_anomaly(anomaly_description, retrieval_query=None, k=5):
    """
    Same retrieve-then-generate mechanics as generator.py's answer_question.
    retrieval_query lets the caller use a cleaner natural-language query for
    the actual search, while still showing the full technical description
    in the prompt/output -- raw telemetry strings make poor search queries.
    """
    query = retrieval_query or anomaly_description
    chunks = retrieve_chunks(query, k=k)

    if not chunks:
        return {"explanation": "I don't have enough information.", "sources": []}

    context = "\n\n".join(c["text"] for c in chunks)
    prompt = RCA_PROMPT_TEMPLATE.format(context=context, anomaly_description=anomaly_description)

    explanation = generate(prompt)
    sources = list({c["source"] for c in chunks})

    return {"explanation": explanation, "sources": sources}


if __name__ == "__main__":
    import os
    from anomaly_detector import load_sample_csvs, detect_anomalies, describe_anomaly, anomaly_to_query

    print("Detecting a real anomaly from telemetry data...")
    df = load_sample_csvs("slice_mixed", max_files=30)
    anomalies = detect_anomalies(df)

    if len(anomalies) == 0:
        print("No anomalies found in this sample -- try a different max_files or threshold.")
    else:
        row = anomalies.iloc[0]
        description = describe_anomaly(row)
        query = anomaly_to_query(row)

        print(f"\nAnomaly: {description}")
        print(f"Retrieval query used: {query}\n")

        result = explain_anomaly(description, retrieval_query=query)
        print(f"Explanation: {result['explanation']}")
        print(f"Sources: {result['sources']}")

    os._exit(0)