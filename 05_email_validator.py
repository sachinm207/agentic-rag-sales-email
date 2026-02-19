import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "us.amazon.nova-pro-v1:0"

VALIDATION_PROMPT = """Review this sales email and check for these issues:

1. PII leaks — does it include personal email addresses, phone numbers, or private info that shouldn't be in a cold outreach?
2. Off-brand promises — does it make guarantees we can't back up (e.g. "guaranteed 10x ROI", specific uptime percentages)?
3. Hallucinated claims — does it reference company details that seem fabricated or unlikely?
4. Tone — is it professional without being pushy or desperate?

Return a JSON object with this structure:
{
  "passed": true/false,
  "issues": ["list of specific issues found, empty if passed"],
  "suggestions": ["optional improvements"]
}

Only return the JSON, nothing else.

Email to validate:
"""


def validate_email(email_text):
    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [{"text": VALIDATION_PROMPT + email_text}]
        }],
        inferenceConfig={"maxTokens": 512, "temperature": 0.0}
    )
    raw = resp["output"]["message"]["content"][0]["text"]
    # try parsing as json
    try:
        # strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}


if __name__ == "__main__":
    # test with a good email
    good_email = """Subject: Scaling your ML infrastructure post-Series C

Hi Priya,

Congrats on DataPulse's Series C — tripling the ML team is an exciting (and challenging) move.

One thing we've seen with fast-growing data science orgs: the infrastructure bottleneck hits hardest
right when you're trying to onboard new people. If your team is spending 60% of their time on infra
instead of modeling, that ratio gets worse before it gets better as you scale.

NeuralForge Studio handles the deployment pipeline so your team can focus on the models. Standardized
model serving, one-click rollbacks, auto-scaling — the stuff that currently requires engineering tickets.

Would a 20-minute demo be worth your time this week?

Best,
Alex"""

    print("=== Good email validation ===")
    result = validate_email(good_email)
    print(json.dumps(result, indent=2))

    # test with a bad email — includes PII and wild promises
    bad_email = """Subject: GUARANTEED 10X ROI with NeuralForge!!

Hi Priya (priya.sharma@datapulse.com, 555-0147),

I know you just closed your round at $85M and your SSN probably looks great right now! LOL

NeuralForge GUARANTEES 99.999% uptime and will reduce your costs by EXACTLY 90% within 30 days
or your money back. We are the #1 AI platform in the world and every Fortune 500 company uses us.

Your competitor FinGuard Analytics is already using us and they said your fraud models are way behind.

Call me NOW at my personal cell 555-0198 before this exclusive offer expires Friday!!

Desperately yours,
Sales Guy"""

    print("\n=== Bad email validation ===")
    result = validate_email(bad_email)
    print(json.dumps(result, indent=2))
