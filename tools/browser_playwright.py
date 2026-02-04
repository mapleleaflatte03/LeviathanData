"""Tool: browser_playwright
Playwright browser automation for web scraping, testing, and interaction.

Supported operations:
- navigate: Go to a URL
- screenshot: Capture page screenshot
- pdf: Generate PDF of page
- content: Get page HTML content
- text: Get visible text content
- click: Click on element
- fill: Fill form fields
- select: Select dropdown options
- evaluate: Execute JavaScript
- wait: Wait for element/condition
"""
from typing import Any, Dict, List, Optional
import json
import asyncio
import base64


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


playwright_sync = None
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    playwright_sync = sync_playwright
except ImportError:
    pass


class BrowserSession:
    """Manages a browser session."""
    
    def __init__(self, browser_type: str = "chromium", headless: bool = True):
        self.browser_type = browser_type
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
    
    def __enter__(self):
        self.playwright = playwright_sync().__enter__()
        
        browser_launcher = getattr(self.playwright, self.browser_type)
        self.browser = browser_launcher.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.page = self.context.new_page()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.__exit__(exc_type, exc_val, exc_tb)


def _navigate(url: str, wait_until: str = "load", timeout: int = 30000, **kwargs) -> Dict[str, Any]:
    """Navigate to a URL and return page info."""
    with BrowserSession(**kwargs) as session:
        response = session.page.goto(url, wait_until=wait_until, timeout=timeout)
        
        return {
            "url": session.page.url,
            "title": session.page.title(),
            "status": response.status if response else None,
        }


def _screenshot(
    url: str,
    output_path: Optional[str] = None,
    full_page: bool = False,
    selector: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Capture screenshot of page or element."""
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        
        screenshot_options = {"full_page": full_page}
        
        if selector:
            element = session.page.locator(selector)
            screenshot_bytes = element.screenshot()
        else:
            screenshot_bytes = session.page.screenshot(**screenshot_options)
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(screenshot_bytes)
            return {"output_path": output_path, "size": len(screenshot_bytes)}
        else:
            return {
                "base64": base64.b64encode(screenshot_bytes).decode("utf-8"),
                "size": len(screenshot_bytes),
            }


def _pdf(url: str, output_path: str, **kwargs) -> Dict[str, Any]:
    """Generate PDF of page."""
    # PDF only works in headless Chromium
    kwargs["browser_type"] = "chromium"
    kwargs["headless"] = True
    
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        
        pdf_bytes = session.page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
        )
        
        return {"output_path": output_path, "size": len(pdf_bytes) if pdf_bytes else 0}


def _get_content(url: str, **kwargs) -> Dict[str, Any]:
    """Get HTML content of page."""
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        
        return {
            "url": session.page.url,
            "title": session.page.title(),
            "html": session.page.content(),
        }


def _get_text(url: str, selector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Get visible text content."""
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        
        if selector:
            elements = session.page.locator(selector).all()
            texts = [el.inner_text() for el in elements]
            return {"texts": texts, "count": len(texts)}
        else:
            text = session.page.locator("body").inner_text()
            return {"text": text}


def _interact(
    url: str,
    actions: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Perform a series of interactions on a page."""
    results = []
    
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        
        for action in actions:
            action_type = action.get("type")
            selector = action.get("selector")
            
            try:
                if action_type == "click":
                    session.page.locator(selector).click()
                    results.append({"action": "click", "selector": selector, "success": True})
                
                elif action_type == "fill":
                    value = action.get("value", "")
                    session.page.locator(selector).fill(value)
                    results.append({"action": "fill", "selector": selector, "success": True})
                
                elif action_type == "select":
                    value = action.get("value")
                    session.page.locator(selector).select_option(value)
                    results.append({"action": "select", "selector": selector, "success": True})
                
                elif action_type == "check":
                    session.page.locator(selector).check()
                    results.append({"action": "check", "selector": selector, "success": True})
                
                elif action_type == "uncheck":
                    session.page.locator(selector).uncheck()
                    results.append({"action": "uncheck", "selector": selector, "success": True})
                
                elif action_type == "hover":
                    session.page.locator(selector).hover()
                    results.append({"action": "hover", "selector": selector, "success": True})
                
                elif action_type == "press":
                    key = action.get("key", "Enter")
                    session.page.locator(selector).press(key)
                    results.append({"action": "press", "selector": selector, "key": key, "success": True})
                
                elif action_type == "wait":
                    timeout = action.get("timeout", 5000)
                    session.page.locator(selector).wait_for(timeout=timeout)
                    results.append({"action": "wait", "selector": selector, "success": True})
                
                elif action_type == "wait_navigation":
                    timeout = action.get("timeout", 30000)
                    session.page.wait_for_load_state("networkidle", timeout=timeout)
                    results.append({"action": "wait_navigation", "success": True})
                
                elif action_type == "screenshot":
                    output_path = action.get("output_path")
                    session.page.screenshot(path=output_path)
                    results.append({"action": "screenshot", "output_path": output_path, "success": True})
                
            except Exception as e:
                results.append({"action": action_type, "selector": selector, "success": False, "error": str(e)})
        
        return {
            "final_url": session.page.url,
            "final_title": session.page.title(),
            "results": results,
        }


def _evaluate(url: str, script: str, **kwargs) -> Dict[str, Any]:
    """Execute JavaScript and return result."""
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        result = session.page.evaluate(script)
        return {"result": result}


def _scrape_table(url: str, selector: str, **kwargs) -> Dict[str, Any]:
    """Scrape table data from page."""
    with BrowserSession(**kwargs) as session:
        session.page.goto(url, wait_until="networkidle")
        
        table = session.page.locator(selector)
        
        # Get headers
        headers = []
        header_cells = table.locator("thead th, tr:first-child th, tr:first-child td").all()
        for cell in header_cells:
            headers.append(cell.inner_text().strip())
        
        # Get rows
        rows = []
        body_rows = table.locator("tbody tr, tr:not(:first-child)").all()
        for row in body_rows:
            cells = row.locator("td, th").all()
            row_data = [cell.inner_text().strip() for cell in cells]
            if row_data and any(row_data):
                rows.append(row_data)
        
        # Convert to records if headers exist
        if headers and len(headers) > 0:
            records = []
            for row in rows:
                if len(row) == len(headers):
                    records.append(dict(zip(headers, row)))
                else:
                    records.append(row)
            return {"headers": headers, "rows": rows, "records": records}
        
        return {"rows": rows}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Playwright browser automation.
    
    Args:
        args: Dictionary with:
            - operation: Action to perform
            - url: Target URL
            - browser: Browser type (chromium, firefox, webkit)
            - headless: Run in headless mode (default: True)
            - selector: CSS/XPath selector for element operations
            - actions: List of actions for interact operation
            - output_path: Path for screenshot/PDF output
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "navigate")
    url = args.get("url", "")
    
    if playwright_sync is None:
        return {
            "tool": "browser_playwright",
            "status": "error",
            "error": "playwright not installed. Run: pip install playwright && playwright install",
        }
    
    browser_kwargs = {
        "browser_type": args.get("browser", "chromium"),
        "headless": args.get("headless", True),
    }
    
    try:
        if operation == "navigate":
            result = _navigate(
                url,
                wait_until=args.get("wait_until", "load"),
                timeout=args.get("timeout", 30000),
                **browser_kwargs
            )
        
        elif operation == "screenshot":
            result = _screenshot(
                url,
                output_path=args.get("output_path"),
                full_page=args.get("full_page", False),
                selector=args.get("selector"),
                **browser_kwargs
            )
        
        elif operation == "pdf":
            result = _pdf(url, args.get("output_path", "page.pdf"), **browser_kwargs)
        
        elif operation == "content":
            result = _get_content(url, **browser_kwargs)
        
        elif operation == "text":
            result = _get_text(url, selector=args.get("selector"), **browser_kwargs)
        
        elif operation == "interact":
            result = _interact(url, actions=args.get("actions", []), **browser_kwargs)
        
        elif operation == "evaluate":
            result = _evaluate(url, script=args.get("script", ""), **browser_kwargs)
        
        elif operation == "scrape_table":
            result = _scrape_table(url, selector=args.get("selector", "table"), **browser_kwargs)
        
        else:
            return {"tool": "browser_playwright", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {
            "tool": "browser_playwright",
            "status": "ok",
            **result,
        }
    
    except Exception as e:
        return {"tool": "browser_playwright", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "screenshot": {
            "operation": "screenshot",
            "url": "https://example.com",
            "output_path": "screenshot.png",
            "full_page": True,
        },
        "scrape_text": {
            "operation": "text",
            "url": "https://example.com",
            "selector": "h1, p",
        },
        "form_fill": {
            "operation": "interact",
            "url": "https://example.com/login",
            "actions": [
                {"type": "fill", "selector": "#username", "value": "user@example.com"},
                {"type": "fill", "selector": "#password", "value": "password123"},
                {"type": "click", "selector": "button[type=submit]"},
                {"type": "wait_navigation"},
            ],
        },
        "scrape_table": {
            "operation": "scrape_table",
            "url": "https://example.com/data",
            "selector": "table.data-table",
        },
    }
