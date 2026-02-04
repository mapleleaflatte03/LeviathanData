"""Tool: viz_plotly_dash
Plotly/Dash visualization generation.

Supported operations:
- line: Create line charts
- bar: Create bar charts
- scatter: Create scatter plots
- pie: Create pie charts
- histogram: Create histograms
- heatmap: Create heatmaps
- box: Create box plots
- to_html: Export chart to HTML
- to_json: Export chart to JSON
"""
from typing import Any, Dict, List, Optional
import json


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


plotly = _optional_import("plotly")
go = None
px = None

if plotly:
    import plotly.graph_objects as go
    import plotly.express as px


def _create_line(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a line chart."""
    fig = go.Figure()
    
    for trace in data.get("traces", []):
        fig.add_trace(go.Scatter(
            x=trace.get("x", []),
            y=trace.get("y", []),
            mode=trace.get("mode", "lines"),
            name=trace.get("name", ""),
            line=trace.get("line", {}),
        ))
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_bar(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a bar chart."""
    fig = go.Figure()
    
    for trace in data.get("traces", []):
        fig.add_trace(go.Bar(
            x=trace.get("x", []),
            y=trace.get("y", []),
            name=trace.get("name", ""),
            marker=trace.get("marker", {}),
        ))
    
    barmode = data.get("barmode", "group")
    fig.update_layout(barmode=barmode)
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_scatter(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a scatter plot."""
    fig = go.Figure()
    
    for trace in data.get("traces", []):
        fig.add_trace(go.Scatter(
            x=trace.get("x", []),
            y=trace.get("y", []),
            mode=trace.get("mode", "markers"),
            name=trace.get("name", ""),
            marker=trace.get("marker", {}),
            text=trace.get("text", []),
        ))
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_pie(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a pie chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=data.get("labels", []),
        values=data.get("values", []),
        hole=data.get("hole", 0),
        textinfo=data.get("textinfo", "percent+label"),
    ))
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_histogram(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a histogram."""
    fig = go.Figure()
    
    for trace in data.get("traces", []):
        fig.add_trace(go.Histogram(
            x=trace.get("x", []),
            nbinsx=trace.get("nbins", None),
            name=trace.get("name", ""),
            opacity=trace.get("opacity", 0.7),
        ))
    
    barmode = data.get("barmode", "overlay")
    fig.update_layout(barmode=barmode)
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_heatmap(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a heatmap."""
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=data.get("z", []),
        x=data.get("x", []),
        y=data.get("y", []),
        colorscale=data.get("colorscale", "Viridis"),
        showscale=data.get("showscale", True),
    ))
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_box(data: Dict[str, Any], layout: Optional[Dict] = None) -> Any:
    """Create a box plot."""
    fig = go.Figure()
    
    for trace in data.get("traces", []):
        fig.add_trace(go.Box(
            y=trace.get("y", []),
            x=trace.get("x", []),
            name=trace.get("name", ""),
            boxpoints=trace.get("boxpoints", "outliers"),
        ))
    
    if layout:
        fig.update_layout(**layout)
    
    return fig


def _create_express(chart_type: str, df_data: List[Dict], **kwargs) -> Any:
    """Create chart using plotly express from data records."""
    pd = _optional_import("pandas")
    if pd is None:
        raise ImportError("pandas is required for express charts")
    
    df = pd.DataFrame(df_data)
    
    express_funcs = {
        "line": px.line,
        "bar": px.bar,
        "scatter": px.scatter,
        "pie": px.pie,
        "histogram": px.histogram,
        "box": px.box,
        "violin": px.violin,
        "area": px.area,
        "treemap": px.treemap,
        "sunburst": px.sunburst,
    }
    
    func = express_funcs.get(chart_type)
    if func is None:
        raise ValueError(f"Unknown express chart type: {chart_type}")
    
    return func(df, **kwargs)


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Plotly visualization operations.
    
    Args:
        args: Dictionary with:
            - operation: Chart type or action
            - data: Chart data (traces, labels, values, etc.)
            - layout: Layout configuration
            - output_path: Path for HTML/JSON output
            - express: If True, use plotly express
            - express_args: Arguments for plotly express
    
    Returns:
        Result dictionary with status and chart data
    """
    args = args or {}
    operation = args.get("operation", "bar")
    data = args.get("data", {})
    layout = args.get("layout", {})
    
    if plotly is None:
        return {"tool": "viz_plotly_dash", "status": "error", "error": "plotly not installed"}
    
    try:
        # Use plotly express if specified
        if args.get("express"):
            df_data = args.get("df_data", [])
            express_args = args.get("express_args", {})
            fig = _create_express(operation, df_data, **express_args)
        else:
            # Use graph_objects
            chart_funcs = {
                "line": _create_line,
                "bar": _create_bar,
                "scatter": _create_scatter,
                "pie": _create_pie,
                "histogram": _create_histogram,
                "heatmap": _create_heatmap,
                "box": _create_box,
            }
            
            func = chart_funcs.get(operation)
            if func is None:
                return {"tool": "viz_plotly_dash", "status": "error", "error": f"Unknown chart type: {operation}"}
            
            fig = func(data, layout)
        
        # Handle output
        output_path = args.get("output_path")
        output_format = args.get("output_format", "html")
        
        result = {
            "tool": "viz_plotly_dash",
            "status": "ok",
            "chart_type": operation,
        }
        
        if output_path:
            if output_format == "html":
                fig.write_html(output_path)
                result["output_path"] = output_path
            elif output_format == "json":
                with open(output_path, "w") as f:
                    f.write(fig.to_json())
                result["output_path"] = output_path
            elif output_format == "png":
                fig.write_image(output_path)
                result["output_path"] = output_path
        
        # Include JSON representation for API response
        result["figure_json"] = json.loads(fig.to_json())
        
        return result
    
    except Exception as e:
        return {"tool": "viz_plotly_dash", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "line_chart": {
            "operation": "line",
            "data": {
                "traces": [
                    {"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25], "name": "Squares"},
                    {"x": [1, 2, 3, 4, 5], "y": [1, 8, 27, 64, 125], "name": "Cubes"},
                ],
            },
            "layout": {"title": "Powers", "xaxis_title": "X", "yaxis_title": "Y"},
        },
        "bar_chart": {
            "operation": "bar",
            "data": {
                "traces": [
                    {"x": ["A", "B", "C"], "y": [10, 20, 15], "name": "Group 1"},
                    {"x": ["A", "B", "C"], "y": [12, 18, 22], "name": "Group 2"},
                ],
                "barmode": "group",
            },
            "layout": {"title": "Comparison"},
        },
        "express_scatter": {
            "operation": "scatter",
            "express": True,
            "df_data": [
                {"x": 1, "y": 2, "category": "A"},
                {"x": 2, "y": 4, "category": "B"},
                {"x": 3, "y": 1, "category": "A"},
            ],
            "express_args": {"x": "x", "y": "y", "color": "category"},
        },
    }
