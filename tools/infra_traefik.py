"""Tool: infra_traefik
Traefik reverse proxy configuration.

Supported operations:
- generate_config: Generate Traefik configuration
- generate_docker_labels: Generate Docker labels for Traefik
- generate_k8s_ingress: Generate Kubernetes IngressRoute
- list_routers: List routers via API
- list_services: List services via API
- list_middlewares: List middlewares via API
"""
from typing import Any, Dict, List, Optional
import json
import yaml


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


requests = _optional_import("requests")


def _generate_static_config(
    entrypoints: Optional[Dict[str, int]] = None,
    dashboard: bool = True,
    api_insecure: bool = True,
    providers: Optional[List[str]] = None,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """Generate Traefik static configuration."""
    entrypoints = entrypoints or {"web": 80, "websecure": 443}
    providers = providers or ["docker"]
    
    config = {
        "api": {
            "dashboard": dashboard,
            "insecure": api_insecure,
        },
        "entryPoints": {
            name: {"address": f":{port}"}
            for name, port in entrypoints.items()
        },
        "log": {
            "level": log_level,
        },
        "providers": {},
    }
    
    if "docker" in providers:
        config["providers"]["docker"] = {
            "exposedByDefault": False,
            "endpoint": "unix:///var/run/docker.sock",
        }
    
    if "file" in providers:
        config["providers"]["file"] = {
            "directory": "/etc/traefik/conf.d",
            "watch": True,
        }
    
    if "kubernetes" in providers:
        config["providers"]["kubernetesIngress"] = {}
    
    return config


def _generate_dynamic_config(
    routers: List[Dict[str, Any]],
    services: Optional[List[Dict[str, Any]]] = None,
    middlewares: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate Traefik dynamic configuration."""
    config = {
        "http": {
            "routers": {},
            "services": {},
            "middlewares": {},
        },
    }
    
    for router in routers:
        name = router.get("name", "default")
        config["http"]["routers"][name] = {
            "rule": router.get("rule", "PathPrefix(`/`)"),
            "service": router.get("service", name),
            "entryPoints": router.get("entrypoints", ["web"]),
        }
        
        if router.get("middlewares"):
            config["http"]["routers"][name]["middlewares"] = router["middlewares"]
        
        if router.get("tls"):
            config["http"]["routers"][name]["tls"] = router["tls"]
    
    if services:
        for service in services:
            name = service.get("name", "default")
            config["http"]["services"][name] = {
                "loadBalancer": {
                    "servers": [
                        {"url": url} for url in service.get("urls", [])
                    ],
                },
            }
    
    if middlewares:
        for mw in middlewares:
            name = mw.get("name", "default")
            mw_type = mw.get("type", "headers")
            config["http"]["middlewares"][name] = {
                mw_type: mw.get("config", {}),
            }
    
    return config


def _generate_docker_labels(
    service_name: str,
    host: str,
    port: int = 80,
    path_prefix: Optional[str] = None,
    middlewares: Optional[List[str]] = None,
    tls: bool = False,
    entrypoint: str = "web",
) -> Dict[str, str]:
    """Generate Docker labels for Traefik."""
    labels = {
        "traefik.enable": "true",
        f"traefik.http.routers.{service_name}.entrypoints": entrypoint,
    }
    
    # Build rule
    rule_parts = [f"Host(`{host}`)"]
    if path_prefix:
        rule_parts.append(f"PathPrefix(`{path_prefix}`)")
    
    labels[f"traefik.http.routers.{service_name}.rule"] = " && ".join(rule_parts)
    
    # Service port
    labels[f"traefik.http.services.{service_name}.loadbalancer.server.port"] = str(port)
    
    # Middlewares
    if middlewares:
        labels[f"traefik.http.routers.{service_name}.middlewares"] = ",".join(middlewares)
    
    # TLS
    if tls:
        labels[f"traefik.http.routers.{service_name}.tls"] = "true"
        labels[f"traefik.http.routers.{service_name}.tls.certresolver"] = "letsencrypt"
    
    return labels


def _generate_k8s_ingress(
    name: str,
    namespace: str,
    host: str,
    service_name: str,
    service_port: int,
    path: str = "/",
    tls: bool = False,
    tls_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate Kubernetes IngressRoute for Traefik."""
    ingress = {
        "apiVersion": "traefik.containo.us/v1alpha1",
        "kind": "IngressRoute",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "entryPoints": ["websecure" if tls else "web"],
            "routes": [
                {
                    "match": f"Host(`{host}`) && PathPrefix(`{path}`)",
                    "kind": "Rule",
                    "services": [
                        {
                            "name": service_name,
                            "port": service_port,
                        },
                    ],
                },
            ],
        },
    }
    
    if tls:
        ingress["spec"]["tls"] = {
            "certResolver": "letsencrypt",
        }
        if tls_secret:
            ingress["spec"]["tls"]["secretName"] = tls_secret
    
    return ingress


def _api_request(
    api_url: str,
    endpoint: str,
) -> Dict[str, Any]:
    """Make request to Traefik API."""
    if requests is None:
        raise ImportError("requests not installed")
    
    url = f"{api_url.rstrip('/')}/api/{endpoint}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Traefik operations."""
    args = args or {}
    operation = args.get("operation", "generate_docker_labels")
    
    try:
        if operation == "generate_static_config":
            config = _generate_static_config(
                entrypoints=args.get("entrypoints"),
                dashboard=args.get("dashboard", True),
                api_insecure=args.get("api_insecure", True),
                providers=args.get("providers"),
                log_level=args.get("log_level", "INFO"),
            )
            return {
                "tool": "infra_traefik",
                "status": "ok",
                "config": config,
                "yaml": yaml.dump(config) if _optional_import("yaml") else None,
            }
        
        elif operation == "generate_dynamic_config":
            config = _generate_dynamic_config(
                routers=args.get("routers", []),
                services=args.get("services"),
                middlewares=args.get("middlewares"),
            )
            return {
                "tool": "infra_traefik",
                "status": "ok",
                "config": config,
            }
        
        elif operation == "generate_docker_labels":
            labels = _generate_docker_labels(
                service_name=args.get("service_name", "app"),
                host=args.get("host", "localhost"),
                port=args.get("port", 80),
                path_prefix=args.get("path_prefix"),
                middlewares=args.get("middlewares"),
                tls=args.get("tls", False),
                entrypoint=args.get("entrypoint", "web"),
            )
            return {"tool": "infra_traefik", "status": "ok", "labels": labels}
        
        elif operation == "generate_k8s_ingress":
            ingress = _generate_k8s_ingress(
                name=args.get("name", "app-ingress"),
                namespace=args.get("namespace", "default"),
                host=args.get("host", "example.com"),
                service_name=args.get("service_name", "app"),
                service_port=args.get("service_port", 80),
                path=args.get("path", "/"),
                tls=args.get("tls", False),
                tls_secret=args.get("tls_secret"),
            )
            return {"tool": "infra_traefik", "status": "ok", "manifest": ingress}
        
        elif operation == "list_routers":
            data = _api_request(args.get("api_url", "http://localhost:8080"), "http/routers")
            return {"tool": "infra_traefik", "status": "ok", "routers": data}
        
        elif operation == "list_services":
            data = _api_request(args.get("api_url", "http://localhost:8080"), "http/services")
            return {"tool": "infra_traefik", "status": "ok", "services": data}
        
        elif operation == "list_middlewares":
            data = _api_request(args.get("api_url", "http://localhost:8080"), "http/middlewares")
            return {"tool": "infra_traefik", "status": "ok", "middlewares": data}
        
        else:
            return {"tool": "infra_traefik", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "infra_traefik", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "docker_labels": {
            "operation": "generate_docker_labels",
            "service_name": "myapp",
            "host": "app.example.com",
            "port": 3000,
            "tls": True,
        },
        "static_config": {
            "operation": "generate_static_config",
            "entrypoints": {"web": 80, "websecure": 443},
            "providers": ["docker", "file"],
        },
        "k8s_ingress": {
            "operation": "generate_k8s_ingress",
            "name": "myapp-ingress",
            "host": "app.example.com",
            "service_name": "myapp",
            "service_port": 80,
            "tls": True,
        },
    }
