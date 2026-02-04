"""Tool: infra_docker
Docker container management and automation.

Supported operations:
- list_containers: List running containers
- list_images: List available images
- run: Run a new container
- stop: Stop a container
- remove: Remove a container
- logs: Get container logs
- exec: Execute command in container
- build: Build image from Dockerfile
- pull: Pull image from registry
- push: Push image to registry
- compose_up: Start docker-compose services
- compose_down: Stop docker-compose services
"""
from typing import Any, Dict, List, Optional
import subprocess
import json
import shutil


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


docker = _optional_import("docker")

# Docker client cache
_docker_client = None


def _get_client():
    """Get Docker client."""
    global _docker_client
    if docker is None:
        raise ImportError("docker SDK not installed. Run: pip install docker")
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def _check_docker_cli() -> bool:
    """Check if docker CLI is available."""
    return shutil.which("docker") is not None


def _run_docker_cmd(args: List[str], capture_output: bool = True) -> Dict[str, Any]:
    """Run docker CLI command."""
    if not _check_docker_cli():
        raise RuntimeError("Docker CLI not found in PATH")
    
    cmd = ["docker"] + args
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout if capture_output else None,
        "stderr": result.stderr if capture_output else None,
    }


def _list_containers(all_containers: bool = False) -> Dict[str, Any]:
    """List Docker containers."""
    try:
        client = _get_client()
        containers = client.containers.list(all=all_containers)
        
        return {
            "containers": [
                {
                    "id": c.short_id,
                    "name": c.name,
                    "image": c.image.tags[0] if c.image.tags else c.image.short_id,
                    "status": c.status,
                    "ports": c.ports,
                    "created": c.attrs.get("Created"),
                }
                for c in containers
            ]
        }
    except Exception:
        # Fallback to CLI
        result = _run_docker_cmd(["ps", "-a" if all_containers else "", "--format", "json"])
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"])
        
        containers = []
        for line in result["stdout"].strip().split("\n"):
            if line:
                containers.append(json.loads(line))
        return {"containers": containers}


def _list_images() -> Dict[str, Any]:
    """List Docker images."""
    try:
        client = _get_client()
        images = client.images.list()
        
        return {
            "images": [
                {
                    "id": img.short_id,
                    "tags": img.tags,
                    "size": img.attrs.get("Size", 0),
                    "created": img.attrs.get("Created"),
                }
                for img in images
            ]
        }
    except Exception:
        result = _run_docker_cmd(["images", "--format", "json"])
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"])
        
        images = []
        for line in result["stdout"].strip().split("\n"):
            if line:
                images.append(json.loads(line))
        return {"images": images}


def _run_container(
    image: str,
    name: Optional[str] = None,
    command: Optional[str] = None,
    ports: Optional[Dict[str, int]] = None,
    volumes: Optional[Dict[str, Dict]] = None,
    environment: Optional[Dict[str, str]] = None,
    detach: bool = True,
    remove: bool = False,
    network: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a new container."""
    try:
        client = _get_client()
        
        kwargs = {
            "image": image,
            "detach": detach,
            "remove": remove,
        }
        
        if name:
            kwargs["name"] = name
        if command:
            kwargs["command"] = command
        if ports:
            kwargs["ports"] = ports
        if volumes:
            kwargs["volumes"] = volumes
        if environment:
            kwargs["environment"] = environment
        if network:
            kwargs["network"] = network
        
        container = client.containers.run(**kwargs)
        
        if detach:
            return {
                "id": container.short_id,
                "name": container.name,
                "status": container.status,
            }
        else:
            return {"output": container.decode("utf-8") if isinstance(container, bytes) else str(container)}
    
    except Exception as e:
        # Fallback to CLI
        args = ["run"]
        if detach:
            args.append("-d")
        if remove:
            args.append("--rm")
        if name:
            args.extend(["--name", name])
        if ports:
            for host_port, container_port in ports.items():
                args.extend(["-p", f"{host_port}:{container_port}"])
        if volumes:
            for host_path, vol_config in volumes.items():
                bind = vol_config.get("bind", "/data")
                mode = vol_config.get("mode", "rw")
                args.extend(["-v", f"{host_path}:{bind}:{mode}"])
        if environment:
            for key, val in environment.items():
                args.extend(["-e", f"{key}={val}"])
        if network:
            args.extend(["--network", network])
        
        args.append(image)
        if command:
            args.append(command)
        
        result = _run_docker_cmd(args)
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"])
        
        return {"id": result["stdout"].strip()[:12]}


def _stop_container(container_id: str, timeout: int = 10) -> Dict[str, Any]:
    """Stop a container."""
    try:
        client = _get_client()
        container = client.containers.get(container_id)
        container.stop(timeout=timeout)
        return {"stopped": container_id}
    except Exception:
        result = _run_docker_cmd(["stop", "-t", str(timeout), container_id])
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"])
        return {"stopped": container_id}


def _remove_container(container_id: str, force: bool = False) -> Dict[str, Any]:
    """Remove a container."""
    try:
        client = _get_client()
        container = client.containers.get(container_id)
        container.remove(force=force)
        return {"removed": container_id}
    except Exception:
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(container_id)
        result = _run_docker_cmd(args)
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"])
        return {"removed": container_id}


def _get_logs(
    container_id: str,
    tail: Optional[int] = None,
    since: Optional[str] = None,
    timestamps: bool = False,
) -> Dict[str, Any]:
    """Get container logs."""
    try:
        client = _get_client()
        container = client.containers.get(container_id)
        
        kwargs = {"timestamps": timestamps}
        if tail:
            kwargs["tail"] = tail
        if since:
            kwargs["since"] = since
        
        logs = container.logs(**kwargs)
        return {"logs": logs.decode("utf-8")}
    
    except Exception:
        args = ["logs"]
        if tail:
            args.extend(["--tail", str(tail)])
        if since:
            args.extend(["--since", since])
        if timestamps:
            args.append("-t")
        args.append(container_id)
        
        result = _run_docker_cmd(args)
        return {"logs": result["stdout"]}


def _exec_command(container_id: str, command: str, workdir: Optional[str] = None) -> Dict[str, Any]:
    """Execute command in container."""
    try:
        client = _get_client()
        container = client.containers.get(container_id)
        
        kwargs = {}
        if workdir:
            kwargs["workdir"] = workdir
        
        exit_code, output = container.exec_run(command, **kwargs)
        return {
            "exit_code": exit_code,
            "output": output.decode("utf-8"),
        }
    except Exception:
        args = ["exec"]
        if workdir:
            args.extend(["-w", workdir])
        args.extend([container_id] + command.split())
        
        result = _run_docker_cmd(args)
        return {
            "exit_code": result["returncode"],
            "output": result["stdout"],
        }


def _build_image(
    path: str,
    tag: str,
    dockerfile: Optional[str] = None,
    buildargs: Optional[Dict[str, str]] = None,
    nocache: bool = False,
) -> Dict[str, Any]:
    """Build Docker image."""
    args = ["build", "-t", tag]
    
    if dockerfile:
        args.extend(["-f", dockerfile])
    if buildargs:
        for key, val in buildargs.items():
            args.extend(["--build-arg", f"{key}={val}"])
    if nocache:
        args.append("--no-cache")
    
    args.append(path)
    
    result = _run_docker_cmd(args)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"])
    
    return {"built": tag, "output": result["stdout"]}


def _pull_image(image: str) -> Dict[str, Any]:
    """Pull image from registry."""
    try:
        client = _get_client()
        image_obj = client.images.pull(image)
        return {"pulled": image, "id": image_obj.short_id}
    except Exception:
        result = _run_docker_cmd(["pull", image])
        if result["returncode"] != 0:
            raise RuntimeError(result["stderr"])
        return {"pulled": image}


def _compose_up(
    compose_file: str = "docker-compose.yml",
    project_name: Optional[str] = None,
    detach: bool = True,
    build: bool = False,
) -> Dict[str, Any]:
    """Start docker-compose services."""
    args = ["compose", "-f", compose_file]
    
    if project_name:
        args.extend(["-p", project_name])
    
    args.append("up")
    
    if detach:
        args.append("-d")
    if build:
        args.append("--build")
    
    result = _run_docker_cmd(args)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"])
    
    return {"status": "started", "output": result["stdout"]}


def _compose_down(
    compose_file: str = "docker-compose.yml",
    project_name: Optional[str] = None,
    volumes: bool = False,
    remove_orphans: bool = False,
) -> Dict[str, Any]:
    """Stop docker-compose services."""
    args = ["compose", "-f", compose_file]
    
    if project_name:
        args.extend(["-p", project_name])
    
    args.append("down")
    
    if volumes:
        args.append("-v")
    if remove_orphans:
        args.append("--remove-orphans")
    
    result = _run_docker_cmd(args)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"])
    
    return {"status": "stopped", "output": result["stdout"]}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Docker operations.
    
    Args:
        args: Dictionary with:
            - operation: Docker operation to perform
            - container_id: Container ID or name
            - image: Image name
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with operation output
    """
    args = args or {}
    operation = args.get("operation", "list_containers")
    
    try:
        if operation == "list_containers":
            result = _list_containers(all_containers=args.get("all", False))
        
        elif operation == "list_images":
            result = _list_images()
        
        elif operation == "run":
            result = _run_container(
                image=args.get("image", ""),
                name=args.get("name"),
                command=args.get("command"),
                ports=args.get("ports"),
                volumes=args.get("volumes"),
                environment=args.get("environment"),
                detach=args.get("detach", True),
                remove=args.get("remove", False),
                network=args.get("network"),
            )
        
        elif operation == "stop":
            result = _stop_container(
                container_id=args.get("container_id", ""),
                timeout=args.get("timeout", 10),
            )
        
        elif operation == "remove":
            result = _remove_container(
                container_id=args.get("container_id", ""),
                force=args.get("force", False),
            )
        
        elif operation == "logs":
            result = _get_logs(
                container_id=args.get("container_id", ""),
                tail=args.get("tail"),
                since=args.get("since"),
                timestamps=args.get("timestamps", False),
            )
        
        elif operation == "exec":
            result = _exec_command(
                container_id=args.get("container_id", ""),
                command=args.get("command", ""),
                workdir=args.get("workdir"),
            )
        
        elif operation == "build":
            result = _build_image(
                path=args.get("path", "."),
                tag=args.get("tag", ""),
                dockerfile=args.get("dockerfile"),
                buildargs=args.get("buildargs"),
                nocache=args.get("nocache", False),
            )
        
        elif operation == "pull":
            result = _pull_image(image=args.get("image", ""))
        
        elif operation == "compose_up":
            result = _compose_up(
                compose_file=args.get("compose_file", "docker-compose.yml"),
                project_name=args.get("project_name"),
                detach=args.get("detach", True),
                build=args.get("build", False),
            )
        
        elif operation == "compose_down":
            result = _compose_down(
                compose_file=args.get("compose_file", "docker-compose.yml"),
                project_name=args.get("project_name"),
                volumes=args.get("volumes", False),
                remove_orphans=args.get("remove_orphans", False),
            )
        
        else:
            return {"tool": "infra_docker", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "infra_docker", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "infra_docker", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "run_container": {
            "operation": "run",
            "image": "nginx:alpine",
            "name": "my-nginx",
            "ports": {"80/tcp": 8080},
            "detach": True,
        },
        "run_with_volumes": {
            "operation": "run",
            "image": "python:3.11-slim",
            "name": "my-python",
            "command": "python -c 'print(\"Hello\")'",
            "volumes": {"/tmp/data": {"bind": "/data", "mode": "rw"}},
            "environment": {"MY_VAR": "value"},
        },
        "build": {
            "operation": "build",
            "path": "./app",
            "tag": "myapp:latest",
            "dockerfile": "Dockerfile",
        },
        "compose_up": {
            "operation": "compose_up",
            "compose_file": "docker-compose.yml",
            "detach": True,
            "build": True,
        },
    }
