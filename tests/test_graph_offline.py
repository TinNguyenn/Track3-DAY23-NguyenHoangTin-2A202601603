"""Offline graph behavior tests using a fake provider boundary."""

from types import SimpleNamespace

import pytest

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class _FakeStructured:
    def invoke(self, messages: object) -> dict[str, str]:
        query = str(messages[-1][1]).lower()  # type: ignore[index]
        if any(word in query for word in ("refund", "delete", "cancel")):
            route, risk = "risky", "high"
        elif any(word in query for word in ("lookup", "status", "search")):
            route, risk = "tool", "low"
        elif any(word in query for word in ("timeout", "failure", "crash")):
            route, risk = "error", "low"
        elif "fix it" in query or "vague" in query:
            route, risk = "missing_info", "low"
        else:
            route, risk = "simple", "low"
        return {"route": route, "risk_level": risk}


class _FakeLLM:
    def with_structured_output(self, schema: object) -> _FakeStructured:
        return _FakeStructured()

    def invoke(self, messages: object) -> SimpleNamespace:
        return SimpleNamespace(content="Grounded fake response from available context.")


@pytest.fixture
def offline_graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nodes, "get_llm", lambda **_: _FakeLLM())
    return build_graph(checkpointer=build_checkpointer("memory"))


@pytest.mark.parametrize(
    ("query", "route"),
    [
        ("How do I reset my password?", Route.SIMPLE),
        ("Lookup order status 123", Route.TOOL),
        ("Can you fix it?", Route.MISSING_INFO),
        ("Refund this customer", Route.RISKY),
        ("Timeout failure while processing", Route.ERROR),
    ],
)
def test_graph_routes_and_finalizes_offline(offline_graph, query: str, route: Route) -> None:
    scenario = Scenario(id=f"offline-{route.value}", query=query, expected_route=route)
    state = initial_state(scenario)
    result = offline_graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )
    assert result["route"] == route.value
    assert result["final_answer"] or result["pending_question"]
    assert result["events"][-1]["node"] == "finalize"


def test_retry_is_bounded_and_dead_letters(offline_graph) -> None:
    scenario = Scenario(
        id="offline-dead-letter",
        query="System failure cannot recover",
        expected_route=Route.ERROR,
        max_attempts=1,
    )
    state = initial_state(scenario)
    result = offline_graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )
    assert result["attempt"] == 1
    assert any(event["node"] == "dead_letter" for event in result["events"])
    assert result["events"][-1]["node"] == "finalize"


def test_sqlite_checkpointer_records_state_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_llm", lambda **_: _FakeLLM())
    checkpointer = build_checkpointer("sqlite", ":memory:")
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="sqlite", query="hello", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    result = graph.invoke(state, config=config)
    assert result["events"][-1]["node"] == "finalize"
    assert list(graph.get_state_history(config))
