"""Tool: infra_prometheus_grafana
Prometheus/Grafana observability stack.

Supported operations:
- prometheus_query: Query Prometheus metrics
- prometheus_query_range: Range query for time series
- prometheus_targets: List scrape targets
- grafana_dashboards: List Grafana dashboards
- grafana_create_dashboard: Create dashboard
- grafana_datasources: List data sources
- generate_prometheus_config: Generate prometheus.yml
- generate_alert_rules: Generate alert rules
"""
from typing import Any, Dict, List, Optional
import json
import time


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


requests = _optional_import("requests")
yaml = _optional_import("yaml")


class PrometheusClient:
    """Prometheus API client."""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip("/")
    
    def query(self, query: str, time_param: Optional[float] = None) -> Dict[str, Any]:
        """Execute instant query."""
        if requests is None:
            raise ImportError("requests not installed")
        
        params = {"query": query}
        if time_param:
            params["time"] = time_param
        
        response = requests.get(f"{self.base_url}/api/v1/query", params=params)
        response.raise_for_status()
        return response.json()
    
    def query_range(
        self,
        query: str,
        start: float,
        end: float,
        step: str = "15s",
    ) -> Dict[str, Any]:
        """Execute range query."""
        if requests is None:
            raise ImportError("requests not installed")
        
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        }
        
        response = requests.get(f"{self.base_url}/api/v1/query_range", params=params)
        response.raise_for_status()
        return response.json()
    
    def targets(self) -> Dict[str, Any]:
        """Get scrape targets."""
        if requests is None:
            raise ImportError("requests not installed")
        
        response = requests.get(f"{self.base_url}/api/v1/targets")
        response.raise_for_status()
        return response.json()
    
    def alerts(self) -> Dict[str, Any]:
        """Get active alerts."""
        if requests is None:
            raise ImportError("requests not installed")
        
        response = requests.get(f"{self.base_url}/api/v1/alerts")
        response.raise_for_status()
        return response.json()


class GrafanaClient:
    """Grafana API client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session() if requests else None
        
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        elif username and password:
            self.session.auth = (username, password)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards."""
        response = self.session.get(f"{self.base_url}/api/search?type=dash-db")
        response.raise_for_status()
        return response.json()
    
    def get_dashboard(self, uid: str) -> Dict[str, Any]:
        """Get dashboard by UID."""
        response = self.session.get(f"{self.base_url}/api/dashboards/uid/{uid}")
        response.raise_for_status()
        return response.json()
    
    def create_dashboard(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update dashboard."""
        payload = {
            "dashboard": dashboard,
            "overwrite": True,
        }
        response = self.session.post(
            f"{self.base_url}/api/dashboards/db",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def list_datasources(self) -> List[Dict[str, Any]]:
        """List data sources."""
        response = self.session.get(f"{self.base_url}/api/datasources")
        response.raise_for_status()
        return response.json()
    
    def create_datasource(self, datasource: Dict[str, Any]) -> Dict[str, Any]:
        """Create data source."""
        response = self.session.post(
            f"{self.base_url}/api/datasources",
            json=datasource,
        )
        response.raise_for_status()
        return response.json()


def _generate_prometheus_config(
    scrape_configs: List[Dict[str, Any]],
    global_config: Optional[Dict[str, Any]] = None,
    alerting: Optional[Dict[str, Any]] = None,
    rule_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate prometheus.yml configuration."""
    config = {
        "global": global_config or {
            "scrape_interval": "15s",
            "evaluation_interval": "15s",
        },
        "scrape_configs": scrape_configs,
    }
    
    if alerting:
        config["alerting"] = alerting
    
    if rule_files:
        config["rule_files"] = rule_files
    
    return config


def _generate_alert_rules(
    groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate Prometheus alert rules."""
    return {"groups": groups}


def _generate_grafana_dashboard(
    title: str,
    panels: List[Dict[str, Any]],
    uid: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate Grafana dashboard JSON."""
    return {
        "uid": uid,
        "title": title,
        "tags": tags or [],
        "timezone": "browser",
        "schemaVersion": 30,
        "panels": panels,
        "editable": True,
        "refresh": "10s",
    }


def _generate_panel(
    title: str,
    panel_type: str,
    query: str,
    datasource: str = "Prometheus",
    grid_pos: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Generate Grafana panel."""
    return {
        "title": title,
        "type": panel_type,
        "gridPos": grid_pos or {"h": 8, "w": 12, "x": 0, "y": 0},
        "datasource": datasource,
        "targets": [
            {
                "expr": query,
                "refId": "A",
            },
        ],
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Prometheus/Grafana operations."""
    args = args or {}
    operation = args.get("operation", "prometheus_query")
    
    try:
        if operation == "prometheus_query":
            client = PrometheusClient(args.get("prometheus_url", "http://localhost:9090"))
            result = client.query(
                query=args.get("query", "up"),
                time_param=args.get("time"),
            )
            return {"tool": "infra_prometheus_grafana", "status": "ok", "result": result}
        
        elif operation == "prometheus_query_range":
            client = PrometheusClient(args.get("prometheus_url", "http://localhost:9090"))
            now = time.time()
            result = client.query_range(
                query=args.get("query", "up"),
                start=args.get("start", now - 3600),
                end=args.get("end", now),
                step=args.get("step", "15s"),
            )
            return {"tool": "infra_prometheus_grafana", "status": "ok", "result": result}
        
        elif operation == "prometheus_targets":
            client = PrometheusClient(args.get("prometheus_url", "http://localhost:9090"))
            result = client.targets()
            return {"tool": "infra_prometheus_grafana", "status": "ok", "targets": result}
        
        elif operation == "prometheus_alerts":
            client = PrometheusClient(args.get("prometheus_url", "http://localhost:9090"))
            result = client.alerts()
            return {"tool": "infra_prometheus_grafana", "status": "ok", "alerts": result}
        
        elif operation == "grafana_dashboards":
            client = GrafanaClient(
                args.get("grafana_url", "http://localhost:3000"),
                api_key=args.get("api_key"),
            )
            dashboards = client.list_dashboards()
            return {"tool": "infra_prometheus_grafana", "status": "ok", "dashboards": dashboards}
        
        elif operation == "grafana_create_dashboard":
            client = GrafanaClient(
                args.get("grafana_url", "http://localhost:3000"),
                api_key=args.get("api_key"),
            )
            result = client.create_dashboard(args.get("dashboard", {}))
            return {"tool": "infra_prometheus_grafana", "status": "ok", "created": result}
        
        elif operation == "generate_prometheus_config":
            config = _generate_prometheus_config(
                scrape_configs=args.get("scrape_configs", []),
                global_config=args.get("global_config"),
                alerting=args.get("alerting"),
                rule_files=args.get("rule_files"),
            )
            yaml_str = yaml.dump(config) if yaml else None
            return {"tool": "infra_prometheus_grafana", "status": "ok", "config": config, "yaml": yaml_str}
        
        elif operation == "generate_alert_rules":
            rules = _generate_alert_rules(groups=args.get("groups", []))
            return {"tool": "infra_prometheus_grafana", "status": "ok", "rules": rules}
        
        elif operation == "generate_dashboard":
            dashboard = _generate_grafana_dashboard(
                title=args.get("title", "Dashboard"),
                panels=args.get("panels", []),
                uid=args.get("uid"),
                tags=args.get("tags"),
            )
            return {"tool": "infra_prometheus_grafana", "status": "ok", "dashboard": dashboard}
        
        else:
            return {"tool": "infra_prometheus_grafana", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "infra_prometheus_grafana", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "prometheus_query": {
            "operation": "prometheus_query",
            "query": "up{job='prometheus'}",
        },
        "generate_config": {
            "operation": "generate_prometheus_config",
            "scrape_configs": [
                {
                    "job_name": "prometheus",
                    "static_configs": [{"targets": ["localhost:9090"]}],
                },
                {
                    "job_name": "node",
                    "static_configs": [{"targets": ["localhost:9100"]}],
                },
            ],
        },
        "generate_dashboard": {
            "operation": "generate_dashboard",
            "title": "System Metrics",
            "panels": [
                {
                    "title": "CPU Usage",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [{"expr": "rate(node_cpu_seconds_total[5m])"}],
                },
            ],
        },
    }
