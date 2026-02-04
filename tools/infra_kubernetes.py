"""Tool: infra_kubernetes
Kubernetes cluster management and automation.

Supported operations:
- list_pods: List pods in namespace
- list_deployments: List deployments
- list_services: List services
- apply: Apply manifest
- delete: Delete resource
- scale: Scale deployment
- logs: Get pod logs
- exec: Execute command in pod
- port_forward: Set up port forwarding
- get_events: Get cluster events
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


kubernetes = _optional_import("kubernetes")


def _check_kubectl() -> bool:
    """Check if kubectl is installed."""
    return shutil.which("kubectl") is not None


def _run_kubectl(args: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
    """Run kubectl command."""
    cmd = ["kubectl"]
    
    if namespace:
        cmd.extend(["-n", namespace])
    
    cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _list_pods(
    namespace: str = "default",
    label_selector: Optional[str] = None,
    all_namespaces: bool = False,
) -> Dict[str, Any]:
    """List pods."""
    args = ["get", "pods", "-o", "json"]
    
    if all_namespaces:
        args.append("--all-namespaces")
        namespace = None
    
    if label_selector:
        args.extend(["-l", label_selector])
    
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        data = json.loads(result["stdout"])
        pods = []
        
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            
            pods.append({
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "phase": status.get("phase"),
                "ready": all(
                    c.get("ready", False)
                    for c in status.get("containerStatuses", [])
                ),
                "restarts": sum(
                    c.get("restartCount", 0)
                    for c in status.get("containerStatuses", [])
                ),
                "ip": status.get("podIP"),
                "node": item.get("spec", {}).get("nodeName"),
            })
        
        return {"pods": pods, "count": len(pods)}
    
    return {"error": result.get("stderr", "Failed to list pods")}


def _list_deployments(
    namespace: str = "default",
    all_namespaces: bool = False,
) -> Dict[str, Any]:
    """List deployments."""
    args = ["get", "deployments", "-o", "json"]
    
    if all_namespaces:
        args.append("--all-namespaces")
        namespace = None
    
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        data = json.loads(result["stdout"])
        deployments = []
        
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            spec = item.get("spec", {})
            status = item.get("status", {})
            
            deployments.append({
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "replicas": spec.get("replicas", 0),
                "ready_replicas": status.get("readyReplicas", 0),
                "available_replicas": status.get("availableReplicas", 0),
                "image": spec.get("template", {}).get("spec", {}).get("containers", [{}])[0].get("image"),
            })
        
        return {"deployments": deployments, "count": len(deployments)}
    
    return {"error": result.get("stderr", "Failed to list deployments")}


def _list_services(
    namespace: str = "default",
    all_namespaces: bool = False,
) -> Dict[str, Any]:
    """List services."""
    args = ["get", "services", "-o", "json"]
    
    if all_namespaces:
        args.append("--all-namespaces")
        namespace = None
    
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        data = json.loads(result["stdout"])
        services = []
        
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            spec = item.get("spec", {})
            
            services.append({
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "type": spec.get("type"),
                "cluster_ip": spec.get("clusterIP"),
                "external_ip": spec.get("externalIPs", []),
                "ports": [
                    {
                        "port": p.get("port"),
                        "target_port": p.get("targetPort"),
                        "protocol": p.get("protocol"),
                    }
                    for p in spec.get("ports", [])
                ],
            })
        
        return {"services": services, "count": len(services)}
    
    return {"error": result.get("stderr", "Failed to list services")}


def _apply(
    manifest: str,
    namespace: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply manifest."""
    if filename:
        args = ["apply", "-f", filename]
    else:
        # Write manifest to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(manifest)
            temp_path = f.name
        
        args = ["apply", "-f", temp_path]
    
    result = _run_kubectl(args, namespace)
    
    if not filename:
        os.unlink(temp_path)
    
    if result["success"]:
        return {"applied": True, "output": result["stdout"]}
    
    return {"error": result.get("stderr", "Apply failed")}


def _delete(
    resource_type: str,
    name: str,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Delete resource."""
    args = ["delete", resource_type, name]
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        return {"deleted": True, "resource": f"{resource_type}/{name}"}
    
    return {"error": result.get("stderr", "Delete failed")}


def _scale(
    deployment: str,
    replicas: int,
    namespace: str = "default",
) -> Dict[str, Any]:
    """Scale deployment."""
    args = ["scale", "deployment", deployment, f"--replicas={replicas}"]
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        return {"scaled": True, "deployment": deployment, "replicas": replicas}
    
    return {"error": result.get("stderr", "Scale failed")}


def _logs(
    pod: str,
    namespace: str = "default",
    container: Optional[str] = None,
    tail: int = 100,
    previous: bool = False,
) -> Dict[str, Any]:
    """Get pod logs."""
    args = ["logs", pod, f"--tail={tail}"]
    
    if container:
        args.extend(["-c", container])
    if previous:
        args.append("--previous")
    
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        return {"logs": result["stdout"], "pod": pod}
    
    return {"error": result.get("stderr", "Failed to get logs")}


def _exec(
    pod: str,
    command: List[str],
    namespace: str = "default",
    container: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute command in pod."""
    args = ["exec", pod, "--"]
    args.extend(command)
    
    if container:
        args.insert(2, "-c")
        args.insert(3, container)
    
    result = _run_kubectl(args, namespace)
    
    return {
        "success": result["success"],
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def _get_events(
    namespace: str = "default",
    all_namespaces: bool = False,
) -> Dict[str, Any]:
    """Get cluster events."""
    args = ["get", "events", "-o", "json", "--sort-by=.lastTimestamp"]
    
    if all_namespaces:
        args.append("--all-namespaces")
        namespace = None
    
    result = _run_kubectl(args, namespace)
    
    if result["success"]:
        data = json.loads(result["stdout"])
        events = []
        
        for item in data.get("items", []):
            events.append({
                "type": item.get("type"),
                "reason": item.get("reason"),
                "message": item.get("message"),
                "object": item.get("involvedObject", {}).get("name"),
                "kind": item.get("involvedObject", {}).get("kind"),
                "count": item.get("count", 1),
                "last_timestamp": item.get("lastTimestamp"),
            })
        
        return {"events": events[-50:], "count": len(events)}
    
    return {"error": result.get("stderr", "Failed to get events")}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Kubernetes operations."""
    args = args or {}
    operation = args.get("operation", "list_pods")
    
    if not _check_kubectl():
        return {
            "tool": "infra_kubernetes",
            "status": "error",
            "error": "kubectl not found. Install kubectl.",
        }
    
    try:
        namespace = args.get("namespace", "default")
        all_ns = args.get("all_namespaces", False)
        
        if operation == "list_pods":
            result = _list_pods(
                namespace=namespace,
                label_selector=args.get("label_selector"),
                all_namespaces=all_ns,
            )
        
        elif operation == "list_deployments":
            result = _list_deployments(namespace=namespace, all_namespaces=all_ns)
        
        elif operation == "list_services":
            result = _list_services(namespace=namespace, all_namespaces=all_ns)
        
        elif operation == "apply":
            result = _apply(
                manifest=args.get("manifest", ""),
                namespace=namespace,
                filename=args.get("filename"),
            )
        
        elif operation == "delete":
            result = _delete(
                resource_type=args.get("resource_type", "pod"),
                name=args.get("name", ""),
                namespace=namespace,
            )
        
        elif operation == "scale":
            result = _scale(
                deployment=args.get("deployment", ""),
                replicas=args.get("replicas", 1),
                namespace=namespace,
            )
        
        elif operation == "logs":
            result = _logs(
                pod=args.get("pod", ""),
                namespace=namespace,
                container=args.get("container"),
                tail=args.get("tail", 100),
                previous=args.get("previous", False),
            )
        
        elif operation == "exec":
            result = _exec(
                pod=args.get("pod", ""),
                command=args.get("command", ["sh"]),
                namespace=namespace,
                container=args.get("container"),
            )
        
        elif operation == "events":
            result = _get_events(namespace=namespace, all_namespaces=all_ns)
        
        else:
            return {"tool": "infra_kubernetes", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "infra_kubernetes", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "infra_kubernetes", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "list_pods": {
            "operation": "list_pods",
            "namespace": "default",
            "label_selector": "app=myapp",
        },
        "scale": {
            "operation": "scale",
            "deployment": "myapp",
            "replicas": 3,
            "namespace": "default",
        },
        "logs": {
            "operation": "logs",
            "pod": "myapp-abc123",
            "tail": 50,
        },
        "apply": {
            "operation": "apply",
            "filename": "deployment.yaml",
        },
    }
