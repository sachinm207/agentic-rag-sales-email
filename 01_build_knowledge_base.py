"""
Build a local vector knowledge base from prospect profiles.

Uses Amazon Titan Text Embeddings V2 to convert each prospect's description
and metadata into a 1024-dimensional vector. Stores everything in a JSON file
so we can do similarity search later without needing OpenSearch or any
managed vector DB.
"""

import json
import boto3
import time
import sys

# Titan Text Embeddings V2 — outputs 1024-dim vectors by default.
# You can request 256 or 512 dims with the "dimensions" param,
# but 1024 gives better retrieval accuracy and costs the same.
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def get_embedding(text):
    """Call Titan Embeddings and return the vector."""
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True   # pre-normalized = cosine similarity is just dot product
        })
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def build_chunks(prospect):
    """
    Turn a prospect profile into searchable text chunks.

    We create multiple chunks per prospect so the retriever can match
    on different aspects — company overview, pain points, tech stack, etc.
    A single giant blob would work but gives worse results for specific queries.
    """
    chunks = []

    # Main description chunk — broadest match
    chunks.append({
        "prospect_id": prospect["id"],
        "chunk_type": "overview",
        "text": f"{prospect['company']} ({prospect['industry']}, {prospect['size']}): {prospect['description']}"
    })

    # Pain points — matches well against "struggling with X" queries
    pain_text = f"{prospect['company']} pain points: " + "; ".join(prospect["pain_points"])
    chunks.append({
        "prospect_id": prospect["id"],
        "chunk_type": "pain_points",
        "text": pain_text
    })

    # Tech stack + ML maturity
    tech_text = (
        f"{prospect['company']} tech stack: {', '.join(prospect['tech_stack'])}. "
        f"ML maturity: {prospect['ml_maturity']}. "
        f"Recent: {prospect['recent_news']}"
    )
    chunks.append({
        "prospect_id": prospect["id"],
        "chunk_type": "tech_profile",
        "text": tech_text
    })

    return chunks


def main():
    with open("prospect_profiles.json") as f:
        data = json.load(f)

    prospects = data["prospects"]
    industry_insights = data["industry_insights"]

    all_chunks = []
    for p in prospects:
        all_chunks.extend(build_chunks(p))

    # Also embed industry insights so the agent can pull sector-level talking points
    for industry, insights in industry_insights.items():
        text = (
            f"Industry: {industry}. "
            f"Trends: {'; '.join(insights['trends'])}. "
            f"Common pain points: {'; '.join(insights['pain_points'])}. "
            f"Talking points for NeuralForge Studio: {'; '.join(insights['talking_points'])}"
        )
        all_chunks.append({
            "prospect_id": f"industry_{industry.replace(' / ', '_').replace(' ', '_')}",
            "chunk_type": "industry_insights",
            "text": text
        })

    print(f"Total chunks to embed: {len(all_chunks)}")

    # Embed each chunk — Titan Embeddings is fast, usually under 100ms per call
    vector_store = []
    start = time.time()
    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk["text"])
        vector_store.append({
            "prospect_id": chunk["prospect_id"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
            "embedding": embedding
        })
        # quick progress indicator
        sys.stdout.write(f"\rEmbedded {i+1}/{len(all_chunks)} chunks")
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\n\nDone. {len(vector_store)} vectors in {elapsed:.1f}s")
    print(f"Average: {elapsed/len(vector_store)*1000:.0f}ms per embedding")
    print(f"Vector dimensions: {len(vector_store[0]['embedding'])}")

    # Save the whole thing
    with open("vector_store.json", "w") as f:
        json.dump(vector_store, f, indent=2)

    print(f"Saved to vector_store.json ({len(vector_store)} entries)")


if __name__ == "__main__":
    main()
