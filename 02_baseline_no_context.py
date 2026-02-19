import json
import boto3
import time

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "us.amazon.nova-pro-v1:0"

def generate_email(company_name):
    start = time.time()
    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [{"text": (
                f"Write a sales email to {company_name} about NeuralForge Studio, "
                "our AI/ML platform for building and deploying models. "
                "Keep it under 200 words, professional but not stiff."
            )}]
        }],
        inferenceConfig={"maxTokens": 512, "temperature": 0.7}
    )
    latency = time.time() - start

    msg = resp["output"]["message"]["content"][0]["text"]
    usage = resp["usage"]
    return msg, usage, latency


if __name__ == "__main__":
    company = "DataPulse Analytics"
    print(f"--- Baseline email to {company} (no context) ---\n")

    email, usage, latency = generate_email(company)
    print(email)
    print(f"\n--- Metrics ---")
    print(f"Input tokens:  {usage['inputTokens']}")
    print(f"Output tokens: {usage['outputTokens']}")
    print(f"Latency:       {latency:.2f}s")

    # Nova Pro pricing: $0.80/1M input, $3.20/1M output
    cost = (usage['inputTokens'] * 0.80 + usage['outputTokens'] * 3.20) / 1_000_000
    print(f"Cost:          ${cost:.6f}")
