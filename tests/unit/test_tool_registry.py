from backend.python.tool_registry import list_tools, run_tool


def test_tool_registry_lists_tools():
    tools = list_tools()
    assert "ml_pandas" in tools


def test_tool_run_returns_payload():
    result = run_tool("ml_pandas", {"sample": True})
    assert result["tool"] == "ml_pandas"
