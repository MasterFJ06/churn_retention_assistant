"""Suggest a retention action from an approved list, optionally using Gemini."""

import json
import os
from typing import Any


ALLOWED_ACTIONS = [
    {
        "action_id": "discounted_12_month_plan",
        "name": "Offer a discounted 12-month plan",
        "description": "Offer a modest discount if the customer is open to an annual commitment.",
    },
    {
        "action_id": "plan_cost_review",
        "name": "Review the customer's plan and monthly cost",
        "description": "Have an agent review available plans and identify a suitable lower-cost option.",
    },
    {
        "action_id": "proactive_support_call",
        "name": "Schedule a proactive support call",
        "description": "Offer a support conversation to discuss service needs and resolve concerns.",
    },
    {
        "action_id": "free_month",
        "name": "Offer one free month",
        "description": "Offer one free month as a short-term retention incentive.",
    },
]


def _fallback_action(customer: dict[str, Any]) -> dict[str, str]:
    """Choose a transparent local suggestion when no LLM key is configured."""
    contract = str(customer.get("Contract", ""))
    monthly_charges = float(customer.get("MonthlyCharges", 0) or 0)

    if contract == "Month-to-month":
        action_id = "discounted_12_month_plan"
    elif monthly_charges >= 80:
        action_id = "plan_cost_review"
    else:
        action_id = "proactive_support_call"

    action = next(item for item in ALLOWED_ACTIONS if item["action_id"] == action_id)
    return {
        "action_id": action["action_id"],
        "action_name": action["name"],
        "recommendation": action["description"],
        "rationale": "Local rule-based fallback; configure GEMINI_API_KEY to use the LLM assistant.",
        "source": "rule-based fallback",
    }


def fallback_recommendation(customer: dict[str, Any]) -> dict[str, str]:
    """Expose the local fallback for app-level API error handling."""
    return _fallback_action(customer)


def generate_recommendation(
    customer: dict[str, Any],
    shap_reasons: list[dict[str, Any]],
    api_key: str | None = None,
) -> dict[str, str]:
    """Return a constrained retention suggestion, using Gemini if configured."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return _fallback_action(customer)

    from google import genai

    client = genai.Client(api_key=key)
    prompt = f"""
You are a careful customer-retention assistant. Use only the supplied facts.
Select exactly one action from ALLOWED_ACTIONS. Never invent discounts, policies,
complaints, outages, or customer facts. SHAP reasons describe model associations,
not proven causes. Do not state the churn score as a guaranteed outcome.

Return a JSON object with exactly these string fields:
action_id, recommendation, rationale, customer_message.

Customer profile (public sample data; contains no customer identifier):
{json.dumps(customer, ensure_ascii=False, default=str)}

Churn risk drivers from the model:
{json.dumps(shap_reasons, ensure_ascii=False, default=str)}

ALLOWED_ACTIONS:
{json.dumps(ALLOWED_ACTIONS, ensure_ascii=False)}

Keep recommendation concise and explain why the selected action fits the facts.
""".strip()

    response_schema = {
        "type": "object",
        "properties": {
            "action_id": {
                "type": "string",
                "enum": [action["action_id"] for action in ALLOWED_ACTIONS],
            },
            "recommendation": {"type": "string"},
            "rationale": {"type": "string"},
            "customer_message": {"type": "string"},
        },
        "required": ["action_id", "recommendation", "rationale", "customer_message"],
        "additionalProperties": False,
    }

    # Google now recommends the Interactions API for current Gemini models.
    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt,
        generation_config={"thinking_level": "minimal"},
        response_format=[
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema,
            }
        ],
        timeout=30,
    )
    result = json.loads(response.output_text or "{}")
    allowed_ids = {action["action_id"] for action in ALLOWED_ACTIONS}
    action_id = result.get("action_id")
    if action_id not in allowed_ids:
        raise ValueError(f"Gemini returned an action outside the approved list: {action_id!r}")

    action = next(item for item in ALLOWED_ACTIONS if item["action_id"] == action_id)
    return {
        "action_id": action_id,
        "action_name": action["name"],
        "recommendation": str(result.get("recommendation", action["description"])),
        "rationale": str(result.get("rationale", "")),
        "customer_message": str(result.get("customer_message", "")),
        "source": "Gemini API",
    }
