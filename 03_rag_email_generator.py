"""
RAG-enhanced email generator.

Retrieves relevant prospect info from the vector store before prompting
the model. Better than baseline because the model gets real context,
but the retrieval is rigid — we decide what to search for up front.
"""

import json
import boto3
import numpy as np
import time

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "us.amazon.nova-pro-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"


def get_embedding(text):
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    )
    return json.loads(resp["body"].read())["embedding"]


def load_vector_store():
    with open("vector_store.json") as f:
        return json.load(f)


def cosine_similarity(a, b):
    # vectors are pre-normalized so dot product = cosine similarity
    return np.dot(a, b)


def search(query, vector_store, top_k=3):
    q_emb = get_embedding(query)
    scored = []
    for entry in vector_store:
        sim = cosine_similarity(q_emb, entry["embedding"])
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def generate_rag_email(company_name, vector_store):
    # retrieve context
    query = f"{company_name} AI ML challenges pain points tech stack"
    results = search(query, vector_store, top_k=3)

    context_block = "\n\n".join([
        f"[Relevance: {score:.3f}] {entry['text']}"
        for score, entry in results
    ])

    start = time.time()
    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [{"text": (
                f"You are writing a sales email to {company_name} about NeuralForge Studio, "
                "our AI/ML platform for model training, deployment, and monitoring.\n\n"
                f"Here is what we know about them:\n{context_block}\n\n"
                "Write a personalized sales email under 200 words. Reference their specific "
                "pain points and recent news. Be professional but conversational."
            )}]
        }],
        inferenceConfig={"maxTokens": 512, "temperature": 0.7}
    )
    latency = time.time() - start

    msg = resp["output"]["message"]["content"][0]["text"]
    usage = resp["usage"]

    return msg, usage, latency, results


if __name__ == "__main__":
    vs = load_vector_store()
    company = "DataPulse Analytics"

    print(f"--- RAG email to {company} ---\n")
    email, usage, latency, retrieval = generate_rag_email(company, vs)

    print("Retrieved context:")
    for score, entry in retrieval:
        print(f"  [{score:.3f}] {entry['chunk_type']} — {entry['prospect_id']}")
    print()
    print(email)
    print(f"\n--- Metrics ---")
    print(f"Input tokens:  {usage['inputTokens']}")
    print(f"Output tokens: {usage['outputTokens']}")
    print(f"Latency:       {latency:.2f}s")

    cost = (usage['inputTokens'] * 0.80 + usage['outputTokens'] * 3.20) / 1_000_000
    print(f"Cost:          ${cost:.6f}")
