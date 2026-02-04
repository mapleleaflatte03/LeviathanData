"""Tool: browser_puppeteer
Puppeteer browser automation via Node.js subprocess.

Supported operations:
- navigate: Navigate to URL
- screenshot: Take screenshot
- pdf: Generate PDF
- evaluate: Execute JavaScript
- click: Click element
- type: Type text
- wait: Wait for selector
- scrape: Extract content
"""
from typing import Any, Dict, List, Optional
import json
import subprocess
import tempfile
import os
import shutil
import base64


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _check_node() -> bool:
    """Check if Node.js is installed."""
    return shutil.which("node") is not None


def _run_puppeteer_script(script: str, timeout: int = 60) -> Dict[str, Any]:
    """Run Puppeteer script via Node.js."""
    if not _check_node():
        raise RuntimeError("Node.js not found")
    
    # Create temp script file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"output": result.stdout}
        else:
            return {"error": result.stderr}
    finally:
        os.unlink(script_path)


def _build_script(
    actions: List[Dict[str, Any]],
    headless: bool = True,
) -> str:
    """Build Puppeteer script."""
    script_lines = [
        "const puppeteer = require('puppeteer');",
        "",
        "(async () => {",
        "  const browser = await puppeteer.launch({",
        f"    headless: {'true' if headless else 'false'},",
        "    args: ['--no-sandbox', '--disable-setuid-sandbox']",
        "  });",
        "  const page = await browser.newPage();",
        "  const results = {};",
        "",
        "  try {",
    ]
    
    for i, action in enumerate(actions):
        action_type = action.get("type", "navigate")
        
        if action_type == "navigate":
            url = action.get("url", "about:blank")
            wait_until = action.get("wait_until", "networkidle2")
            script_lines.append(f"    await page.goto('{url}', {{ waitUntil: '{wait_until}' }});")
        
        elif action_type == "screenshot":
            path = action.get("path", f"/tmp/screenshot_{i}.png")
            full_page = action.get("full_page", False)
            script_lines.append(f"    const screenshot{i} = await page.screenshot({{")
            script_lines.append(f"      path: '{path}',")
            script_lines.append(f"      fullPage: {str(full_page).lower()}")
            script_lines.append("    });")
            script_lines.append(f"    results.screenshot{i} = '{path}';")
        
        elif action_type == "pdf":
            path = action.get("path", f"/tmp/page_{i}.pdf")
            script_lines.append(f"    await page.pdf({{ path: '{path}', format: 'A4' }});")
            script_lines.append(f"    results.pdf{i} = '{path}';")
        
        elif action_type == "evaluate":
            js_code = action.get("script", "document.title")
            script_lines.append(f"    results.eval{i} = await page.evaluate(() => {{ return {js_code}; }});")
        
        elif action_type == "click":
            selector = action.get("selector", "body")
            script_lines.append(f"    await page.click('{selector}');")
        
        elif action_type == "type":
            selector = action.get("selector", "input")
            text = action.get("text", "")
            script_lines.append(f"    await page.type('{selector}', '{text}');")
        
        elif action_type == "wait":
            selector = action.get("selector")
            if selector:
                timeout = action.get("timeout", 30000)
                script_lines.append(f"    await page.waitForSelector('{selector}', {{ timeout: {timeout} }});")
            else:
                ms = action.get("ms", 1000)
                script_lines.append(f"    await new Promise(r => setTimeout(r, {ms}));")
        
        elif action_type == "scrape":
            selector = action.get("selector", "body")
            attr = action.get("attribute", "innerText")
            script_lines.append(f"    results.scrape{i} = await page.$$eval('{selector}', els => els.map(e => e.{attr}));")
        
        elif action_type == "content":
            script_lines.append(f"    results.content = await page.content();")
        
        elif action_type == "cookies":
            script_lines.append(f"    results.cookies = await page.cookies();")
        
        elif action_type == "set_viewport":
            width = action.get("width", 1920)
            height = action.get("height", 1080)
            script_lines.append(f"    await page.setViewport({{ width: {width}, height: {height} }});")
    
    script_lines.extend([
        "    console.log(JSON.stringify(results));",
        "  } catch (error) {",
        "    console.log(JSON.stringify({ error: error.message }));",
        "  } finally {",
        "    await browser.close();",
        "  }",
        "})();",
    ])
    
    return "\n".join(script_lines)


def _navigate(url: str, **kwargs) -> Dict[str, Any]:
    """Navigate to URL and get content."""
    actions = [
        {"type": "navigate", "url": url},
        {"type": "content"},
    ]
    
    if kwargs.get("screenshot"):
        actions.append({"type": "screenshot", "path": kwargs["screenshot"]})
    
    script = _build_script(actions, headless=kwargs.get("headless", True))
    return _run_puppeteer_script(script)


def _screenshot(url: str, path: str, **kwargs) -> Dict[str, Any]:
    """Take screenshot of URL."""
    actions = [
        {"type": "navigate", "url": url},
        {
            "type": "screenshot",
            "path": path,
            "full_page": kwargs.get("full_page", False),
        },
    ]
    
    script = _build_script(actions, headless=True)
    return _run_puppeteer_script(script)


def _pdf(url: str, path: str, **kwargs) -> Dict[str, Any]:
    """Generate PDF from URL."""
    actions = [
        {"type": "navigate", "url": url},
        {"type": "pdf", "path": path},
    ]
    
    script = _build_script(actions, headless=True)
    return _run_puppeteer_script(script)


def _scrape(url: str, selectors: Dict[str, str], **kwargs) -> Dict[str, Any]:
    """Scrape content from URL."""
    actions = [{"type": "navigate", "url": url}]
    
    for name, selector in selectors.items():
        actions.append({
            "type": "scrape",
            "selector": selector,
            "attribute": kwargs.get("attribute", "innerText"),
        })
    
    script = _build_script(actions, headless=True)
    return _run_puppeteer_script(script)


def _execute(url: str, script: str, **kwargs) -> Dict[str, Any]:
    """Execute JavaScript on page."""
    actions = [
        {"type": "navigate", "url": url},
        {"type": "evaluate", "script": script},
    ]
    
    script_code = _build_script(actions, headless=True)
    return _run_puppeteer_script(script_code)


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Puppeteer operations."""
    args = args or {}
    operation = args.get("operation", "navigate")
    
    if not _check_node():
        return {
            "tool": "browser_puppeteer",
            "status": "error",
            "error": "Node.js not found. Install Node.js and puppeteer.",
        }
    
    try:
        if operation == "navigate":
            result = _navigate(
                url=args.get("url", "https://example.com"),
                screenshot=args.get("screenshot"),
                headless=args.get("headless", True),
            )
        
        elif operation == "screenshot":
            result = _screenshot(
                url=args.get("url", "https://example.com"),
                path=args.get("path", "/tmp/screenshot.png"),
                full_page=args.get("full_page", False),
            )
        
        elif operation == "pdf":
            result = _pdf(
                url=args.get("url", "https://example.com"),
                path=args.get("path", "/tmp/page.pdf"),
            )
        
        elif operation == "scrape":
            result = _scrape(
                url=args.get("url", "https://example.com"),
                selectors=args.get("selectors", {"title": "h1"}),
            )
        
        elif operation == "execute":
            result = _execute(
                url=args.get("url", "https://example.com"),
                script=args.get("script", "document.title"),
            )
        
        elif operation == "custom":
            actions = args.get("actions", [])
            script = _build_script(actions, headless=args.get("headless", True))
            result = _run_puppeteer_script(script, timeout=args.get("timeout", 60))
        
        else:
            return {"tool": "browser_puppeteer", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "browser_puppeteer", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "browser_puppeteer", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "navigate": {
            "operation": "navigate",
            "url": "https://example.com",
        },
        "screenshot": {
            "operation": "screenshot",
            "url": "https://example.com",
            "path": "/tmp/screenshot.png",
            "full_page": True,
        },
        "scrape": {
            "operation": "scrape",
            "url": "https://news.ycombinator.com",
            "selectors": {
                "titles": ".titleline a",
            },
        },
        "custom": {
            "operation": "custom",
            "actions": [
                {"type": "navigate", "url": "https://google.com"},
                {"type": "type", "selector": "textarea[name=q]", "text": "puppeteer"},
                {"type": "wait", "ms": 1000},
                {"type": "screenshot", "path": "/tmp/google.png"},
            ],
        },
    }
