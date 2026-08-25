"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel

from .llm import get_llm
from .state import AgentState, make_event


class Classification(BaseModel):
    """The constrained response schema used by the classifier."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "medium", "high"]


class Evaluation(BaseModel):
    """Constrained LLM-as-judge response for tool quality."""

    verdict: Literal["success", "needs_retry"]
    rationale: str


def _message_content(response: object) -> str:
    """Normalize LangChain text and content-block responses."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    return str(content).strip()


def _llm_error(node: str, exc: Exception) -> dict[str, Any]:
    """Convert provider/configuration failures into an auditable state update."""
    message = f"{type(exc).__name__}: {exc}"
    return {
        "errors": [message],
        "events": [make_event(node, "failed", "LLM call failed", error=message)],
    }


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    system_prompt = """
You classify customer-support requests into exactly one route.
Return structured output only. Apply this priority when more than one intent is
present: risky > tool > missing_info > error > simple.

risky: side effects such as refunds, deletion, cancellation, account changes, or
sending messages. tool: a factual lookup or search requiring a backend tool.
missing_info: vague or incomplete requests that cannot be acted on safely.
error: an explicit timeout, crash, outage, or failed system operation.
simple: a general informational question answerable without a tool or side effect.
Choose high risk_level only for risky, medium for potentially consequential
requests, and low otherwise.
""".strip()
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(Classification)
        response = structured_llm.invoke(
            [("system", system_prompt), ("human", f"Support request:\n{query}")]
        )
        result = (
            response
            if isinstance(response, Classification)
            else Classification.model_validate(response)
        )
        return {
            "route": result.route,
            "risk_level": result.risk_level,
            "messages": [f"classification:{result.route}"],
            "events": [
                make_event(
                    "classify",
                    "completed",
                    "query classified",
                    route=result.route,
                    risk_level=result.risk_level,
                )
            ],
        }
    except Exception as exc:
        # A missing key/provider must be visible and must not leave the graph
        # suspended. Route the failed classification through the bounded error path.
        return {
            "route": "error",
            "risk_level": "unknown",
            "messages": ["classification:error"],
            **_llm_error("classify", exc),
        }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = max(0, int(state.get("attempt", 0)))
    route = state.get("route", "")
    if (route == "error" or state.get("should_retry", False)) and attempt < 2:
        result = f"ERROR: transient backend failure on attempt {attempt + 1}"
        return {
            "tool_results": [result],
            "errors": [result],
            "events": [make_event("tool", "failed", "transient tool failure", attempt=attempt)],
        }

    result = f"SUCCESS: mock lookup completed for request: {state.get('query', '')}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", "tool result available", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", []) or []
    latest = str(results[-1]) if results else ""
    heuristic = "needs_retry" if not latest or "ERROR" in latest.upper() else "success"
    evaluation = heuristic
    judge = "heuristic"
    judge_verdict: str | None = None

    # Deterministic safety checks take precedence over the judge: an explicit
    # backend error must be retried, while an empty result cannot be trusted.
    if heuristic == "success":
        try:
            judge_llm = get_llm(temperature=0.0).with_structured_output(Evaluation)
            response = judge_llm.invoke(
                [
                    (
                        "system",
                        "Evaluate whether this support tool result is complete and usable. "
                        "Return needs_retry for missing, ambiguous, or failed results; "
                        "return success only when it directly supports answering the request.",
                    ),
                    (
                        "human",
                        json.dumps(
                            {"query": state.get("query", ""), "tool_result": latest},
                            ensure_ascii=False,
                        ),
                    ),
                ]
            )
            result = (
                response
                if isinstance(response, Evaluation)
                else Evaluation.model_validate(response)
            )
            judge_verdict = result.verdict
            # The mock tool's SUCCESS contract is authoritative. The judge is
            # still recorded for observability, but cannot create a false retry
            # when the tool explicitly reports a complete result.
            if not latest.startswith("SUCCESS:"):
                evaluation = result.verdict
            judge = "llm"
        except Exception:
            # Evaluation is a quality enhancement, not a new failure mode.
            # Fall back to the deterministic gate if the provider is unavailable.
            pass
    return {
        "evaluation_result": evaluation,
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation=evaluation,
                judge=judge,
                judge_verdict=judge_verdict,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    context = {
        "query": state.get("query", ""),
        "route": state.get("route", ""),
        "tool_results": list(state.get("tool_results", []) or []),
        "approval": state.get("approval"),
        "proposed_action": state.get("proposed_action"),
        "errors": list(state.get("errors", []) or []),
    }
    system_prompt = """
You are a support agent. Write a concise, helpful response to the user.
Ground every factual claim in the supplied context. Do not invent order data,
account changes, approvals, or tool results. If the context reports a failure,
explain the limitation and give a safe next step. If a risky action was approved,
confirm only that the approved workflow completed; do not claim an external
side effect beyond the tool result.
""".strip()
    try:
        response = get_llm(temperature=0.2).invoke(
            [("system", system_prompt), ("human", json.dumps(context, ensure_ascii=False))]
        )
        answer = _message_content(response)
        if not answer:
            raise ValueError("LLM returned an empty answer")
        return {
            "final_answer": answer,
            "events": [make_event("answer", "completed", "grounded answer generated")],
        }
    except Exception as exc:
        failure = _llm_error("answer", exc)
        return {
            **failure,
            "final_answer": (
                "I could not generate a response because the language model is unavailable."
            ),
        }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    approval = state.get("approval") or {}
    if approval and not approval.get("approved", False):
        question = "What alternative outcome would you like instead of the unapproved action?"
    else:
        question = (
            "Could you provide the specific account, order, or task details needed to help with "
            f"\"{query}\"?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    proposed_action = (
        f"Proposed action: {query}. This request may change customer data or "
        "communicate externally "
        "and requires explicit approval before execution."
    )
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "completed", "action prepared for approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return an approval decision and an audit event.
    """
    proposed_action = state.get("proposed_action") or state.get("query", "")
    decision: Any = {
        "approved": True,
        "reviewer": "mock-reviewer",
        "comment": "approved for lab run",
    }
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() in {"1", "true", "yes"}:
        try:
            from langgraph.types import interrupt

            decision = interrupt({"type": "approval_request", "action": proposed_action})
        except ImportError as exc:
            return {**_llm_error("approval", exc), "approval": {"approved": False}}
    if isinstance(decision, bool):
        decision = {"approved": decision}
    if not isinstance(decision, dict):
        decision = {"approved": False, "comment": "invalid approval response"}
    approval = {
        "approved": bool(decision.get("approved", False)),
        "reviewer": str(decision.get("reviewer", "human-reviewer")),
        "comment": str(decision.get("comment", "")),
    }
    return {
        "approval": approval,
        "events": [
            make_event(
                "approval",
                "completed",
                "approval decision recorded",
                approved=approval["approved"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = max(0, int(state.get("attempt", 0))) + 1
    maximum = max(0, int(state.get("max_attempts", 3)))
    message = f"Transient failure; retry attempt {attempt} of {maximum}."
    return {
        "attempt": attempt,
        "errors": [message],
        "events": [make_event("retry", "completed", "retry attempt recorded", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempts = int(state.get("attempt", 0))
    maximum = int(state.get("max_attempts", 3))
    answer = (
        "I could not complete this request after "
        f"{attempts} attempt(s). The issue has been recorded for support follow-up."
    )
    return {
        "final_answer": answer,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "request moved to dead-letter handling",
                attempts=attempts,
                max_attempts=maximum,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
