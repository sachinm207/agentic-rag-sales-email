"""
Agentic RAG email personalizer.

Instead of us deciding what to retrieve, we give the model three tools
and let IT decide what information it needs. The model calls tools in a
loop until it's satisfied, then writes the email.

This is the core pattern: Converse API + toolConfig + manual tool dispatch loop.
No Bedrock Agents needed — we control the whole thing.
"""

import json
import boto3
import numpy as np
import time
import re

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "us.amazon.nova-pro-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"


# --- Vector store helpers ---

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


def load_profiles():
    with open("prospect_profiles.json") as f:
        return json.load(f)


VECTOR_STORE = load_vector_store()
PROFILES_DATA = load_profiles()


# --- Tool implementations ---

def tool_search_knowledge_base(query, top_k=3):
    """Semantic search over the prospect knowledge base."""
    q_emb = get_embedding(query)
    scored = []
    for entry in VECTOR_STORE:
        sim = float(np.dot(q_emb, entry["embedding"]))
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, entry in scored[:top_k]:
        results.append({
            "relevance_score": round(score, 4),
            "prospect_id": entry["prospect_id"],
            "chunk_type": entry["chunk_type"],
            "text": entry["text"]
        })
    return {"results": results}


def tool_get_prospect_profile(prospect_id):
    """Full profile lookup by ID."""
    for p in PROFILES_DATA["prospects"]:
        if p["id"] == prospect_id:
            # return everything except embedding-related fields
            return {
                "company": p["company"],
                "industry": p["industry"],
                "size": p["size"],
                "ml_maturity": p["ml_maturity"],
                "pain_points": p["pain_points"],
                "tech_stack": p["tech_stack"],
                "recent_news": p["recent_news"],
                "key_contact": p["key_contact"],
                "description": p["description"]
            }
    return {"error": f"No prospect found with id '{prospect_id}'"}


def tool_search_industry_insights(industry):
    """Get industry-specific trends, pain points, and talking points."""
    insights = PROFILES_DATA.get("industry_insights", {})
    if industry in insights:
        return insights[industry]
    # fuzzy match — try partial
    for key, val in insights.items():
        if industry.lower() in key.lower():
            return val
    return {"error": f"No insights found for industry '{industry}'"}


# Map tool names to functions
TOOL_DISPATCH = {
    "search_knowledge_base": lambda params: tool_search_knowledge_base(params["query"]),
    "get_prospect_profile": lambda params: tool_get_prospect_profile(params["prospect_id"]),
    "search_industry_insights": lambda params: tool_search_industry_insights(params["industry"]),
}


# --- Tool config for Converse API ---

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "search_knowledge_base",
                "description": (
                    "Search the prospect knowledge base using a natural language query. "
                    "Returns the top 3 most relevant chunks with relevance scores. "
                    "Use this to find information about specific companies, industries, or pain points."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_prospect_profile",
                "description": (
                    "Retrieve the full profile for a specific prospect by their ID. "
                    "Returns company details, pain points, tech stack, recent news, and key contact. "
                    "Use this after search_knowledge_base identifies a relevant prospect."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "prospect_id": {
                                "type": "string",
                                "description": "The prospect's unique identifier (e.g. 'datapulse', 'medvault')"
                            }
                        },
                        "required": ["prospect_id"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "search_industry_insights",
                "description": (
                    "Get industry-specific trends, common pain points, and NeuralForge Studio talking points. "
                    "Use this to tailor the email with industry-relevant messaging."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "industry": {
                                "type": "string",
                                "description": "Industry name (e.g. 'fintech', 'healthcare', 'manufacturing')"
                            }
                        },
                        "required": ["industry"]
                    }
                }
            }
        }
    ]
}


def run_agentic_loop(company_name, max_turns=10):
    """
    The main agentic loop.

    Send the initial prompt to Converse with toolConfig. If the model
    returns tool_use blocks, execute them and feed results back.
    Keep going until the model gives a final text response (stopReason = "end_turn").
    """
    messages = [{
        "role": "user",
        "content": [{"text": (
            f"Write a personalized sales email to {company_name} about NeuralForge Studio, "
            "our AI/ML platform for model training, deployment, and monitoring.\n\n"
            "You have access to tools that let you research the prospect. Use them to find "
            "specific details about the company — their pain points, tech stack, recent news, "
            "and industry trends. Then write a compelling, personalized email under 200 words.\n\n"
            "Be specific. Reference actual details you find. Don't write generic filler."
        )}]
    }]

    trace = []  # record every step for debugging/blog
    total_input_tokens = 0
    total_output_tokens = 0
    loop_start = time.time()
    turn = 0

    while turn < max_turns:
        turn += 1
        step_start = time.time()

        resp = bedrock.converse(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 1024, "temperature": 0.7}
        )

        step_latency = time.time() - step_start
        usage = resp["usage"]
        total_input_tokens += usage["inputTokens"]
        total_output_tokens += usage["outputTokens"]
        stop_reason = resp["stopReason"]

        assistant_msg = resp["output"]["message"]
        messages.append(assistant_msg)

        # Process each content block in the response
        tool_results = []
        for block in assistant_msg["content"]:
            if "text" in block:
                clean = re.sub(r"<thinking>.*?</thinking>\s*", "", block["text"], flags=re.DOTALL).strip()
                if clean:
                    trace.append({
                        "turn": turn,
                        "type": "text",
                        "content": clean[:200] + "..." if len(clean) > 200 else clean,
                        "latency": f"{step_latency:.2f}s"
                    })
            elif "toolUse" in block:
                tool_call = block["toolUse"]
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_id = tool_call["toolUseId"]

                trace.append({
                    "turn": turn,
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_input,
                    "latency": f"{step_latency:.2f}s"
                })

                # Execute the tool
                exec_start = time.time()
                if tool_name in TOOL_DISPATCH:
                    result = TOOL_DISPATCH[tool_name](tool_input)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
                exec_time = time.time() - exec_start

                trace.append({
                    "turn": turn,
                    "type": "tool_result",
                    "tool": tool_name,
                    "exec_time": f"{exec_time:.2f}s"
                })

                tool_results.append({
                    "toolUseId": tool_id,
                    "content": [{"json": result}]
                })

        # If there were tool calls, send results back
        if tool_results:
            messages.append({
                "role": "user",
                "content": [{"toolResult": tr} for tr in tool_results]
            })

        # If the model is done (no more tool calls), break
        if stop_reason == "end_turn":
            break

    total_latency = time.time() - loop_start

    # extract final email text
    final_text = ""
    for block in messages[-1]["content"] if messages[-1]["role"] == "assistant" else []:
        if "text" in block:
            final_text = block["text"]

    # if last message was a user (tool result), check the one before
    if not final_text:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                for block in msg["content"]:
                    if "text" in block:
                        final_text = block["text"]
                        break
                if final_text:
                    break

    # Nova models sometimes wrap reasoning in <thinking> tags — strip them
    final_text = re.sub(r"<thinking>.*?</thinking>\s*", "", final_text, flags=re.DOTALL)

    return {
        "email": final_text,
        "trace": trace,
        "turns": turn,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_latency": total_latency
    }


if __name__ == "__main__":
    company = "DataPulse Analytics"
    print(f"--- Agentic RAG email to {company} ---\n")

    result = run_agentic_loop(company)

    print("=== AGENT TRACE ===")
    for step in result["trace"]:
        if step["type"] == "tool_call":
            print(f"  Turn {step['turn']}: {step['tool']}({json.dumps(step['input'])})")
        elif step["type"] == "tool_result":
            print(f"           → executed in {step['exec_time']}")
        elif step["type"] == "text":
            preview = step["content"][:100] + "..." if len(step["content"]) > 100 else step["content"]
            print(f"  Turn {step['turn']}: [text] {preview}")
    print()

    print("=== GENERATED EMAIL ===")
    print(result["email"])
    print()

    print("=== METRICS ===")
    print(f"Turns:         {result['turns']}")
    print(f"Input tokens:  {result['total_input_tokens']}")
    print(f"Output tokens: {result['total_output_tokens']}")
    print(f"Total latency: {result['total_latency']:.2f}s")

    cost = (result['total_input_tokens'] * 0.80 + result['total_output_tokens'] * 3.20) / 1_000_000
    print(f"Cost:          ${cost:.6f}")
