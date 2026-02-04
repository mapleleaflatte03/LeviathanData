"""Tool: viz_evidence
Evidence.dev for generating static data reports.

Supported operations:
- create_project: Create a new Evidence project
- create_page: Create a markdown report page
- add_query: Add SQL query to page
- build: Build static site
- dev: Start development server
"""
from typing import Any, Dict, List, Optional
import json
import subprocess
import shutil
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _check_npm() -> bool:
    """Check if npm is installed."""
    return shutil.which("npm") is not None


def _check_npx() -> bool:
    """Check if npx is installed."""
    return shutil.which("npx") is not None


def _run_command(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run a command."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _create_project(
    project_path: str,
    project_name: str = "evidence-report",
) -> Dict[str, Any]:
    """Create a new Evidence project."""
    if not _check_npx():
        raise RuntimeError("npx not found. Install Node.js.")
    
    os.makedirs(project_path, exist_ok=True)
    
    cmd = ["npx", "degit", "evidence-dev/template", project_name]
    result = _run_command(cmd, cwd=project_path)
    
    if result["success"]:
        full_path = os.path.join(project_path, project_name)
        
        # Install dependencies
        install_result = _run_command(["npm", "install"], cwd=full_path)
        
        return {
            "created": True,
            "path": full_path,
            "deps_installed": install_result["success"],
        }
    
    return {"error": result.get("stderr", "Failed to create project")}


def _create_page(
    project_path: str,
    page_name: str,
    title: str,
    content: str,
    queries: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Create a markdown report page."""
    pages_dir = os.path.join(project_path, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    # Build markdown content
    md_lines = [
        "---",
        f"title: {title}",
        "---",
        "",
    ]
    
    # Add queries if provided
    if queries:
        for query in queries:
            query_name = query.get("name", "query")
            sql = query.get("sql", "")
            md_lines.extend([
                f"```sql {query_name}",
                sql,
                "```",
                "",
            ])
    
    # Add content
    md_lines.append(content)
    
    page_path = os.path.join(pages_dir, f"{page_name}.md")
    
    with open(page_path, "w") as f:
        f.write("\n".join(md_lines))
    
    return {
        "created": True,
        "page_path": page_path,
        "page_name": page_name,
    }


def _add_source(
    project_path: str,
    source_name: str,
    source_type: str = "sqlite",
    connection_string: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a data source to the project."""
    sources_dir = os.path.join(project_path, "sources", source_name)
    os.makedirs(sources_dir, exist_ok=True)
    
    # Create connection.yaml
    connection = {
        "name": source_name,
        "type": source_type,
    }
    
    if connection_string:
        connection["connection_string"] = connection_string
    
    if config:
        connection.update(config)
    
    connection_path = os.path.join(sources_dir, "connection.yaml")
    
    # Write as YAML-like format
    with open(connection_path, "w") as f:
        for key, value in connection.items():
            f.write(f"{key}: {value}\n")
    
    return {
        "added": True,
        "source_name": source_name,
        "source_type": source_type,
        "path": connection_path,
    }


def _build(project_path: str) -> Dict[str, Any]:
    """Build static site."""
    if not _check_npm():
        raise RuntimeError("npm not found")
    
    result = _run_command(["npm", "run", "build"], cwd=project_path)
    
    if result["success"]:
        build_dir = os.path.join(project_path, "build")
        return {
            "built": True,
            "build_dir": build_dir,
            "exists": os.path.exists(build_dir),
        }
    
    return {"error": result.get("stderr", "Build failed")}


def _dev(
    project_path: str,
    port: int = 3000,
) -> Dict[str, Any]:
    """Start development server."""
    if not _check_npm():
        raise RuntimeError("npm not found")
    
    # Start in background
    process = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(port)],
        cwd=project_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    return {
        "started": True,
        "pid": process.pid,
        "url": f"http://localhost:{port}",
    }


def _generate_chart_component(
    chart_type: str,
    data_query: str,
    x: str,
    y: str,
    title: Optional[str] = None,
) -> str:
    """Generate Evidence chart component markdown."""
    components = {
        "line": f"<LineChart data={{{{ {data_query} }}}} x={x} y={y} />",
        "bar": f"<BarChart data={{{{ {data_query} }}}} x={x} y={y} />",
        "area": f"<AreaChart data={{{{ {data_query} }}}} x={x} y={y} />",
        "scatter": f"<ScatterPlot data={{{{ {data_query} }}}} x={x} y={y} />",
        "pie": f"<PieChart data={{{{ {data_query} }}}} name={x} value={y} />",
        "table": f"<DataTable data={{{{ {data_query} }}}} />",
        "value": f"<Value data={{{{ {data_query} }}}} column={y} />",
    }
    
    component = components.get(chart_type, components["table"])
    
    if title:
        return f"## {title}\n\n{component}"
    return component


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Evidence operations."""
    args = args or {}
    operation = args.get("operation", "create_page")
    
    try:
        if operation == "create_project":
            result = _create_project(
                project_path=args.get("project_path", "."),
                project_name=args.get("project_name", "evidence-report"),
            )
        
        elif operation == "create_page":
            result = _create_page(
                project_path=args.get("project_path", "."),
                page_name=args.get("page_name", "index"),
                title=args.get("title", "Report"),
                content=args.get("content", ""),
                queries=args.get("queries"),
            )
        
        elif operation == "add_source":
            result = _add_source(
                project_path=args.get("project_path", "."),
                source_name=args.get("source_name", "data"),
                source_type=args.get("source_type", "sqlite"),
                connection_string=args.get("connection_string"),
                config=args.get("config"),
            )
        
        elif operation == "build":
            result = _build(project_path=args.get("project_path", "."))
        
        elif operation == "dev":
            result = _dev(
                project_path=args.get("project_path", "."),
                port=args.get("port", 3000),
            )
        
        elif operation == "generate_chart":
            chart_md = _generate_chart_component(
                chart_type=args.get("chart_type", "line"),
                data_query=args.get("data_query", "query_results"),
                x=args.get("x", "x"),
                y=args.get("y", "y"),
                title=args.get("title"),
            )
            result = {"markdown": chart_md}
        
        else:
            return {"tool": "viz_evidence", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "viz_evidence", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "viz_evidence", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "create_project": {
            "operation": "create_project",
            "project_path": "/tmp",
            "project_name": "my-report",
        },
        "create_page": {
            "operation": "create_page",
            "project_path": "./my-report",
            "page_name": "sales",
            "title": "Sales Report",
            "content": "## Overview\n\nThis report shows sales data.",
            "queries": [
                {"name": "sales_data", "sql": "SELECT * FROM sales"},
            ],
        },
        "generate_chart": {
            "operation": "generate_chart",
            "chart_type": "line",
            "data_query": "sales_data",
            "x": "date",
            "y": "revenue",
            "title": "Revenue Over Time",
        },
    }
