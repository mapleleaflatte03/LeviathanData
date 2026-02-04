"""Tool: viz_streamlit
Streamlit app generation and management.

Supported operations:
- create_app: Generate Streamlit app code
- run_app: Start a Streamlit app
- stop_app: Stop a running app
- list_running: List running apps
"""
from typing import Any, Dict, List, Optional
import json
import subprocess
import os
import signal


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


# Process cache for running apps
_running_apps: Dict[str, subprocess.Popen] = {}


def _create_app(
    title: str,
    components: List[Dict[str, Any]],
    output_path: str,
    theme: str = "dark",
) -> Dict[str, Any]:
    """Generate Streamlit app code."""
    
    code_lines = [
        "import streamlit as st",
        "import pandas as pd",
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "",
        f'st.set_page_config(page_title="{title}", layout="wide")',
        "",
        f'st.title("{title}")',
        "",
    ]
    
    for component in components:
        comp_type = component.get("type", "text")
        
        if comp_type == "text":
            text = component.get("content", "")
            code_lines.append(f'st.write("""{text}""")')
        
        elif comp_type == "header":
            text = component.get("content", "")
            level = component.get("level", 2)
            if level == 1:
                code_lines.append(f'st.header("{text}")')
            elif level == 2:
                code_lines.append(f'st.subheader("{text}")')
            else:
                code_lines.append(f'st.markdown("### {text}")')
        
        elif comp_type == "dataframe":
            var_name = component.get("var_name", "df")
            code_lines.append(f"st.dataframe({var_name})")
        
        elif comp_type == "chart":
            chart_type = component.get("chart_type", "line")
            data_var = component.get("data_var", "df")
            x = component.get("x", "x")
            y = component.get("y", "y")
            
            if chart_type == "line":
                code_lines.append(f'fig = px.line({data_var}, x="{x}", y="{y}")')
            elif chart_type == "bar":
                code_lines.append(f'fig = px.bar({data_var}, x="{x}", y="{y}")')
            elif chart_type == "scatter":
                code_lines.append(f'fig = px.scatter({data_var}, x="{x}", y="{y}")')
            elif chart_type == "pie":
                code_lines.append(f'fig = px.pie({data_var}, names="{x}", values="{y}")')
            
            code_lines.append("st.plotly_chart(fig, use_container_width=True)")
        
        elif comp_type == "input":
            input_type = component.get("input_type", "text")
            label = component.get("label", "Input")
            var_name = component.get("var_name", "input_value")
            
            if input_type == "text":
                code_lines.append(f'{var_name} = st.text_input("{label}")')
            elif input_type == "number":
                code_lines.append(f'{var_name} = st.number_input("{label}")')
            elif input_type == "slider":
                min_val = component.get("min", 0)
                max_val = component.get("max", 100)
                code_lines.append(f'{var_name} = st.slider("{label}", {min_val}, {max_val})')
            elif input_type == "selectbox":
                options = component.get("options", [])
                code_lines.append(f'{var_name} = st.selectbox("{label}", {options})')
        
        elif comp_type == "button":
            label = component.get("label", "Click")
            code_lines.append(f'if st.button("{label}"):')
            code_lines.append('    st.success("Button clicked!")')
        
        elif comp_type == "file_upload":
            label = component.get("label", "Upload file")
            var_name = component.get("var_name", "uploaded_file")
            code_lines.append(f'{var_name} = st.file_uploader("{label}")')
        
        elif comp_type == "sidebar":
            code_lines.append("with st.sidebar:")
            sidebar_content = component.get("content", "Sidebar")
            code_lines.append(f'    st.write("{sidebar_content}")')
        
        elif comp_type == "columns":
            num_cols = component.get("num", 2)
            code_lines.append(f"cols = st.columns({num_cols})")
        
        elif comp_type == "metric":
            label = component.get("label", "Metric")
            value = component.get("value", 0)
            delta = component.get("delta")
            if delta:
                code_lines.append(f'st.metric("{label}", {value}, delta={delta})')
            else:
                code_lines.append(f'st.metric("{label}", {value})')
        
        elif comp_type == "code":
            code = component.get("content", "")
            lang = component.get("language", "python")
            code_lines.append(f'st.code("""{code}""", language="{lang}")')
        
        code_lines.append("")
    
    code = "\n".join(code_lines)
    
    with open(output_path, "w") as f:
        f.write(code)
    
    return {
        "output_path": output_path,
        "created": True,
        "num_components": len(components),
        "code_length": len(code),
    }


def _run_app(
    app_path: str,
    port: int = 8501,
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a Streamlit app."""
    app_id = app_id or os.path.basename(app_path)
    
    if app_id in _running_apps:
        return {"error": f"App '{app_id}' is already running"}
    
    cmd = [
        "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    _running_apps[app_id] = process
    
    return {
        "app_id": app_id,
        "port": port,
        "url": f"http://localhost:{port}",
        "pid": process.pid,
        "running": True,
    }


def _stop_app(app_id: str) -> Dict[str, Any]:
    """Stop a running app."""
    if app_id not in _running_apps:
        return {"error": f"App '{app_id}' is not running"}
    
    process = _running_apps[app_id]
    process.terminate()
    
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    
    del _running_apps[app_id]
    
    return {"app_id": app_id, "stopped": True}


def _list_running() -> Dict[str, Any]:
    """List running apps."""
    apps = []
    
    for app_id, process in _running_apps.items():
        apps.append({
            "app_id": app_id,
            "pid": process.pid,
            "running": process.poll() is None,
        })
    
    return {"apps": apps, "count": len(apps)}


def _get_template(template_name: str) -> Dict[str, Any]:
    """Get a pre-built app template."""
    templates = {
        "dashboard": {
            "title": "Data Dashboard",
            "components": [
                {"type": "header", "content": "Key Metrics", "level": 2},
                {"type": "metric", "label": "Total Users", "value": 1000, "delta": 50},
                {"type": "metric", "label": "Revenue", "value": "$10,000", "delta": "$500"},
                {"type": "header", "content": "Data Visualization", "level": 2},
                {"type": "chart", "chart_type": "line", "data_var": "df", "x": "date", "y": "value"},
            ],
        },
        "data_explorer": {
            "title": "Data Explorer",
            "components": [
                {"type": "file_upload", "label": "Upload CSV", "var_name": "uploaded_file"},
                {"type": "dataframe", "var_name": "df"},
                {"type": "chart", "chart_type": "scatter", "data_var": "df", "x": "x", "y": "y"},
            ],
        },
        "form": {
            "title": "Input Form",
            "components": [
                {"type": "input", "input_type": "text", "label": "Name", "var_name": "name"},
                {"type": "input", "input_type": "number", "label": "Age", "var_name": "age"},
                {"type": "input", "input_type": "selectbox", "label": "Category", "var_name": "cat", "options": ["A", "B", "C"]},
                {"type": "button", "label": "Submit"},
            ],
        },
    }
    
    if template_name in templates:
        return {"template": templates[template_name]}
    
    return {"error": f"Unknown template: {template_name}", "available": list(templates.keys())}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Streamlit operations."""
    args = args or {}
    operation = args.get("operation", "create_app")
    
    try:
        if operation == "create_app":
            result = _create_app(
                title=args.get("title", "Streamlit App"),
                components=args.get("components", []),
                output_path=args.get("output_path", "app.py"),
                theme=args.get("theme", "dark"),
            )
        
        elif operation == "run_app":
            result = _run_app(
                app_path=args.get("app_path", "app.py"),
                port=args.get("port", 8501),
                app_id=args.get("app_id"),
            )
        
        elif operation == "stop_app":
            result = _stop_app(app_id=args.get("app_id", ""))
        
        elif operation == "list_running":
            result = _list_running()
        
        elif operation == "get_template":
            result = _get_template(template_name=args.get("template", "dashboard"))
        
        else:
            return {"tool": "viz_streamlit", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "viz_streamlit", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "viz_streamlit", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_app": {
            "operation": "create_app",
            "title": "My Dashboard",
            "output_path": "dashboard.py",
            "components": [
                {"type": "header", "content": "Welcome", "level": 1},
                {"type": "text", "content": "This is a sample dashboard."},
                {"type": "metric", "label": "Users", "value": 100, "delta": 10},
                {"type": "input", "input_type": "slider", "label": "Value", "var_name": "val", "min": 0, "max": 100},
            ],
        },
        "run_app": {
            "operation": "run_app",
            "app_path": "dashboard.py",
            "port": 8501,
        },
        "get_template": {
            "operation": "get_template",
            "template": "dashboard",
        },
    }
