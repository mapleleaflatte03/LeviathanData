"""Tool: infra_caddy
Caddy reverse proxy configuration.

Supported operations:
- generate_caddyfile: Generate Caddyfile configuration
- generate_json_config: Generate JSON configuration
- validate_config: Validate Caddyfile
- reload: Reload Caddy configuration
- list_routes: List current routes via API
"""
from typing import Any, Dict, List, Optional
import json
import subprocess
import shutil
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


requests = _optional_import("requests")


def _check_caddy() -> bool:
    """Check if caddy is installed."""
    return shutil.which("caddy") is not None


def _generate_caddyfile(
    sites: List[Dict[str, Any]],
    global_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate Caddyfile configuration."""
    lines = []
    
    # Global options block
    if global_options:
        lines.append("{")
        if global_options.get("email"):
            lines.append(f"    email {global_options['email']}")
        if global_options.get("admin_off"):
            lines.append("    admin off")
        if global_options.get("auto_https"):
            lines.append(f"    auto_https {global_options['auto_https']}")
        if global_options.get("log_level"):
            lines.append(f"    log {{ level {global_options['log_level']} }}")
        lines.append("}")
        lines.append("")
    
    # Site blocks
    for site in sites:
        host = site.get("host", "localhost")
        port = site.get("port")
        
        # Build address
        if port:
            address = f"{host}:{port}"
        else:
            address = host
        
        lines.append(address + " {")
        
        # Reverse proxy
        if site.get("reverse_proxy"):
            upstream = site["reverse_proxy"]
            if isinstance(upstream, str):
                lines.append(f"    reverse_proxy {upstream}")
            elif isinstance(upstream, dict):
                target = upstream.get("to", "localhost:8080")
                path = upstream.get("path", "")
                if path:
                    lines.append(f"    reverse_proxy {path} {target}")
                else:
                    lines.append(f"    reverse_proxy {target}")
                
                # Health check
                if upstream.get("health_check"):
                    lines.append(f"        health_uri {upstream['health_check']}")
        
        # Static file server
        if site.get("file_server"):
            fs_config = site["file_server"]
            if isinstance(fs_config, bool):
                lines.append("    file_server")
            elif isinstance(fs_config, dict):
                root = fs_config.get("root", "/var/www")
                lines.append(f"    root * {root}")
                if fs_config.get("browse"):
                    lines.append("    file_server browse")
                else:
                    lines.append("    file_server")
        
        # TLS
        if site.get("tls"):
            tls_config = site["tls"]
            if isinstance(tls_config, str):
                lines.append(f"    tls {tls_config}")
            elif isinstance(tls_config, dict):
                cert = tls_config.get("cert", "")
                key = tls_config.get("key", "")
                if cert and key:
                    lines.append(f"    tls {cert} {key}")
                elif tls_config.get("internal"):
                    lines.append("    tls internal")
        
        # Headers
        if site.get("headers"):
            lines.append("    header {")
            for header, value in site["headers"].items():
                lines.append(f"        {header} {value}")
            lines.append("    }")
        
        # Logging
        if site.get("log"):
            log_config = site["log"]
            if isinstance(log_config, bool):
                lines.append("    log")
            elif isinstance(log_config, dict):
                output = log_config.get("output", "stdout")
                lines.append(f"    log {{ output {output} }}")
        
        # Encode (gzip)
        if site.get("encode"):
            lines.append("    encode gzip zstd")
        
        # Custom directives
        if site.get("directives"):
            for directive in site["directives"]:
                lines.append(f"    {directive}")
        
        lines.append("}")
        lines.append("")
    
    return "\n".join(lines)


def _generate_json_config(
    sites: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate Caddy JSON configuration."""
    routes = []
    
    for site in sites:
        host = site.get("host", "localhost")
        
        route = {
            "match": [{"host": [host]}],
            "handle": [],
        }
        
        if site.get("reverse_proxy"):
            upstream = site["reverse_proxy"]
            if isinstance(upstream, str):
                upstreams = [{"dial": upstream}]
            else:
                upstreams = [{"dial": upstream.get("to", "localhost:8080")}]
            
            route["handle"].append({
                "handler": "reverse_proxy",
                "upstreams": upstreams,
            })
        
        if site.get("file_server"):
            route["handle"].append({
                "handler": "file_server",
                "root": site.get("root", "/var/www"),
            })
        
        routes.append(route)
    
    return {
        "apps": {
            "http": {
                "servers": {
                    "main": {
                        "listen": [":443", ":80"],
                        "routes": routes,
                    },
                },
            },
        },
    }


def _validate_config(caddyfile_path: str) -> Dict[str, Any]:
    """Validate Caddyfile."""
    if not _check_caddy():
        raise RuntimeError("caddy not found")
    
    result = subprocess.run(
        ["caddy", "validate", "--config", caddyfile_path],
        capture_output=True,
        text=True,
    )
    
    return {
        "valid": result.returncode == 0,
        "output": result.stdout + result.stderr,
    }


def _reload(admin_url: str = "http://localhost:2019") -> Dict[str, Any]:
    """Reload Caddy configuration."""
    if requests is None:
        raise ImportError("requests not installed")
    
    response = requests.post(f"{admin_url}/load")
    
    return {
        "reloaded": response.status_code == 200,
        "status_code": response.status_code,
    }


def _list_routes(admin_url: str = "http://localhost:2019") -> Dict[str, Any]:
    """List current routes via API."""
    if requests is None:
        raise ImportError("requests not installed")
    
    response = requests.get(f"{admin_url}/config/")
    
    if response.status_code == 200:
        return {"config": response.json()}
    
    return {"error": f"Failed with status {response.status_code}"}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Caddy operations."""
    args = args or {}
    operation = args.get("operation", "generate_caddyfile")
    
    try:
        if operation == "generate_caddyfile":
            caddyfile = _generate_caddyfile(
                sites=args.get("sites", []),
                global_options=args.get("global_options"),
            )
            return {"tool": "infra_caddy", "status": "ok", "caddyfile": caddyfile}
        
        elif operation == "generate_json_config":
            config = _generate_json_config(sites=args.get("sites", []))
            return {"tool": "infra_caddy", "status": "ok", "config": config}
        
        elif operation == "validate":
            result = _validate_config(caddyfile_path=args.get("caddyfile_path", "Caddyfile"))
            return {"tool": "infra_caddy", "status": "ok", **result}
        
        elif operation == "reload":
            result = _reload(admin_url=args.get("admin_url", "http://localhost:2019"))
            return {"tool": "infra_caddy", "status": "ok", **result}
        
        elif operation == "list_routes":
            result = _list_routes(admin_url=args.get("admin_url", "http://localhost:2019"))
            return {"tool": "infra_caddy", "status": "ok", **result}
        
        else:
            return {"tool": "infra_caddy", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "infra_caddy", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "generate_caddyfile": {
            "operation": "generate_caddyfile",
            "global_options": {
                "email": "admin@example.com",
            },
            "sites": [
                {
                    "host": "app.example.com",
                    "reverse_proxy": "localhost:3000",
                    "encode": True,
                },
                {
                    "host": "static.example.com",
                    "file_server": {
                        "root": "/var/www/static",
                        "browse": True,
                    },
                },
            ],
        },
        "reverse_proxy_config": {
            "operation": "generate_caddyfile",
            "sites": [
                {
                    "host": "api.example.com",
                    "reverse_proxy": {
                        "to": "localhost:8080",
                        "health_check": "/health",
                    },
                    "headers": {
                        "X-Frame-Options": "DENY",
                        "X-Content-Type-Options": "nosniff",
                    },
                },
            ],
        },
    }
