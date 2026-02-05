"""
OpenClaw Professional Report Generator

Generates real PDF/HTML reports from data and charts, NOT UI screenshots.
Uses WeasyPrint for PDF generation and Plotly for chart rendering.

Features:
- Executive summary with KPIs
- Data tables with proper formatting
- Interactive charts rendered to static images
- Analysis and recommendations
- Professional styling
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("report_generator")

# Check available libraries
WEASYPRINT_AVAILABLE = False
PLOTLY_AVAILABLE = False
KALEIDO_AVAILABLE = False

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    pass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    pass

# Output directory
REPORTS_DIR = Path("/root/leviathan/data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """
    Professional report generator for OpenClaw.
    
    Creates real PDF/HTML reports from structured data and chart configurations.
    """
    
    def __init__(
        self,
        language: str = "vi",
        company_name: str = "",
        report_title: str = "",
    ):
        self.language = language
        self.company_name = company_name
        self.report_title = report_title or self._default_title()
        self.charts: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.kpis: Dict[str, Any] = {}
        self.analysis_text: str = ""
        self.recommendations: List[str] = []
        self.osint_data: Dict[str, Any] = {}
    
    def _default_title(self) -> str:
        """Generate default report title."""
        if self.language == "vi":
            return f"Báo Cáo Phân Tích OSINT - {self.company_name}"
        return f"OSINT Analysis Report - {self.company_name}"
    
    def set_kpis(self, kpis: Dict[str, Any]) -> "ReportGenerator":
        """Set KPIs for the report."""
        self.kpis = kpis
        return self
    
    def set_analysis(self, analysis_text: str) -> "ReportGenerator":
        """Set main analysis text."""
        self.analysis_text = analysis_text
        return self
    
    def set_recommendations(self, recommendations: List[str]) -> "ReportGenerator":
        """Set recommendations list."""
        self.recommendations = recommendations
        return self
    
    def set_osint_data(self, osint_data: Dict[str, Any]) -> "ReportGenerator":
        """Set OSINT findings data."""
        self.osint_data = osint_data
        return self
    
    def add_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        title: str = "",
        description: str = "",
    ) -> "ReportGenerator":
        """Add a chart to the report."""
        self.charts.append({
            "type": chart_type,
            "data": data,
            "title": title,
            "description": description,
        })
        return self
    
    def add_table(
        self,
        data: List[Dict[str, Any]],
        title: str = "",
        columns: Optional[List[str]] = None,
    ) -> "ReportGenerator":
        """Add a data table to the report."""
        self.tables.append({
            "data": data,
            "title": title,
            "columns": columns or (list(data[0].keys()) if data else []),
        })
        return self
    
    def _render_svg_bar_chart(self, data: Dict, title: str, colors: List[str] = None) -> str:
        """Render a simple SVG bar chart (no external dependencies)."""
        x_vals = data.get("x", [])
        y_vals = data.get("y", [])
        if not x_vals or not y_vals:
            return ""
        
        # Convert values to numbers, skip non-numeric
        numeric_y = []
        for v in y_vals:
            try:
                numeric_y.append(float(v) if isinstance(v, (int, float)) else 0)
            except (ValueError, TypeError):
                numeric_y.append(0)
        y_vals = numeric_y
        
        colors = colors or ["#117A65", "#1A5276", "#F39C12", "#E74C3C", "#27AE60", "#8E44AD"]
        max_val = max(y_vals) if y_vals else 1
        if max_val <= 0:
            max_val = 1
        bar_width = 60
        spacing = 20
        chart_height = 200
        chart_width = len(x_vals) * (bar_width + spacing) + 60
        
        svg = f'''<svg width="{chart_width}" height="{chart_height + 60}" xmlns="http://www.w3.org/2000/svg">
            <style>
                .bar-label {{ font-family: Segoe UI, Arial; font-size: 10px; fill: #666; }}
                .bar-value {{ font-family: Segoe UI, Arial; font-size: 11px; fill: #333; font-weight: bold; }}
                .chart-title {{ font-family: Segoe UI, Arial; font-size: 12px; fill: #2C3E50; font-weight: 600; }}
            </style>
            <text x="{chart_width/2}" y="15" text-anchor="middle" class="chart-title">{title}</text>
'''
        for i, (label, value) in enumerate(zip(x_vals, y_vals)):
            x = 40 + i * (bar_width + spacing)
            bar_height = (value / max_val) * chart_height
            y = chart_height + 25 - bar_height
            color = colors[i % len(colors)]
            label_str = str(label)[:10] if label else ""
            
            svg += f'''
            <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="4" />
            <text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" class="bar-value">{value:.0f}</text>
            <text x="{x + bar_width/2}" y="{chart_height + 45}" text-anchor="middle" class="bar-label">{label_str}</text>
'''
        svg += '</svg>'
        return svg
    
    def _render_svg_pie_chart(self, data: Dict, title: str, colors: List[str] = None) -> str:
        """Render a simple SVG pie/donut chart."""
        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            return ""
        
        # Convert values to numbers
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v) if isinstance(v, (int, float)) else 0)
            except (ValueError, TypeError):
                numeric_values.append(0)
        values = numeric_values
        
        colors = colors or ["#27AE60", "#E74C3C", "#F39C12", "#1A5276", "#8E44AD"]
        total = sum(values) if values else 1
        if total <= 0:
            total = 1
        cx, cy, r = 120, 120, 80
        inner_r = 50  # Donut hole
        
        svg = f'''<svg width="280" height="280" xmlns="http://www.w3.org/2000/svg">
            <style>
                .pie-label {{ font-family: Segoe UI, Arial; font-size: 10px; fill: #333; }}
                .chart-title {{ font-family: Segoe UI, Arial; font-size: 12px; fill: #2C3E50; font-weight: 600; }}
            </style>
            <text x="140" y="15" text-anchor="middle" class="chart-title">{title}</text>
'''
        import math
        start_angle = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            angle = (value / total) * 360
            end_angle = start_angle + angle
            
            # SVG arc path
            start_rad = math.radians(start_angle - 90)
            end_rad = math.radians(end_angle - 90)
            
            x1 = cx + r * math.cos(start_rad)
            y1 = cy + r * math.sin(start_rad)
            x2 = cx + r * math.cos(end_rad)
            y2 = cy + r * math.sin(end_rad)
            
            x1_inner = cx + inner_r * math.cos(start_rad)
            y1_inner = cy + inner_r * math.sin(start_rad)
            x2_inner = cx + inner_r * math.cos(end_rad)
            y2_inner = cy + inner_r * math.sin(end_rad)
            
            large_arc = 1 if angle > 180 else 0
            color = colors[i % len(colors)]
            
            path = f"M {x1_inner},{y1_inner} L {x1},{y1} A {r},{r} 0 {large_arc},1 {x2},{y2} L {x2_inner},{y2_inner} A {inner_r},{inner_r} 0 {large_arc},0 {x1_inner},{y1_inner}"
            svg += f'<path d="{path}" fill="{color}" />'
            
            # Legend
            legend_y = 250 + i * 0  # Skip legend for now
            start_angle = end_angle
        
        # Legend below
        for i, (label, value) in enumerate(zip(labels, values)):
            pct = (value / total * 100) if total > 0 else 0
            color = colors[i % len(colors)]
            lx = 30 + i * 120
            svg += f'''
            <rect x="{lx}" y="250" width="12" height="12" fill="{color}" rx="2" />
            <text x="{lx + 16}" y="260" class="pie-label">{label}: {pct:.0f}%</text>
'''
        
        svg += '</svg>'
        return svg
    
    def _render_chart_to_base64(
        self,
        chart_config: Dict[str, Any],
    ) -> str:
        """Render a Plotly chart to base64 PNG image."""
        if not PLOTLY_AVAILABLE:
            return ""
        
        chart_type = chart_config.get("type", "bar")
        data = chart_config.get("data", {})
        title = chart_config.get("title", "")
        
        try:
            if chart_type == "bar":
                fig = go.Figure(data=[
                    go.Bar(
                        x=data.get("x", []),
                        y=data.get("y", []),
                        marker_color=["#117A65", "#1A5276", "#F39C12", "#E74C3C", "#8E44AD"][:len(data.get("x", []))],
                        text=data.get("y", []),
                        textposition="outside",
                    )
                ])
            elif chart_type == "barh":
                # Horizontal bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=data.get("x", []),
                        y=data.get("y", []),
                        orientation='h',
                        marker_color=["#117A65", "#1A5276", "#F39C12", "#27AE60", "#E74C3C"][:len(data.get("y", []))],
                        text=[f"{v:.0f}ms" for v in data.get("x", [])],
                        textposition="outside",
                    )
                ])
            elif chart_type == "line":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data.get("x", []),
                        y=data.get("y", []),
                        mode="lines+markers",
                        line=dict(color="#117A65", width=3),
                        marker=dict(size=8),
                    )
                ])
            elif chart_type == "pie":
                fig = go.Figure(data=[
                    go.Pie(
                        labels=data.get("labels", []),
                        values=data.get("values", []),
                        hole=0.4,  # Donut style
                        marker=dict(colors=["#27AE60", "#E74C3C", "#F39C12", "#1A5276"]),
                        textinfo="label+percent",
                        textfont=dict(size=12),
                    )
                ])
            elif chart_type == "heatmap":
                fig = go.Figure(data=[
                    go.Heatmap(
                        z=data.get("z", [[]]),
                        x=data.get("x", []),
                        y=data.get("y", []),
                        colorscale="Teal"
                    )
                ])
            elif chart_type == "scatter":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data.get("x", []),
                        y=data.get("y", []),
                        mode="markers",
                        marker=dict(color="#117A65", size=12)
                    )
                ])
            elif chart_type == "gauge":
                # Risk gauge chart
                value = data.get("value", 50)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={"text": title},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#117A65"},
                        "steps": [
                            {"range": [0, 33], "color": "#d4edda"},
                            {"range": [33, 66], "color": "#fff3cd"},
                            {"range": [66, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "#E74C3C", "width": 4},
                            "thickness": 0.75,
                            "value": 80,
                        },
                    },
                ))
            else:
                # Default to bar
                fig = go.Figure(data=[
                    go.Bar(x=data.get("x", []), y=data.get("y", []), marker_color="#117A65")
                ])
            
            fig.update_layout(
                title=dict(text=title, font=dict(size=14, color="#2C3E50")),
                paper_bgcolor="white",
                plot_bgcolor="#fafafa",
                font=dict(family="Segoe UI, Arial", size=11, color="#2C3E50"),
                margin=dict(l=60, r=40, t=60, b=50),
                showlegend=chart_type == "pie",
            )
            
            # Add gridlines for bar/line charts
            if chart_type in ["bar", "barh", "line"]:
                fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
                fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
            
            # Convert to base64 image
            if KALEIDO_AVAILABLE:
                try:
                    img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
                    return base64.b64encode(img_bytes).decode("utf-8")
                except Exception as kaleido_err:
                    logger.warning(f"Kaleido image export failed: {kaleido_err}")
                    # Fall through to SVG fallback below
            
            # SVG fallback when Kaleido not available or fails
            logger.info(f"Using SVG fallback for chart: {chart_type}")
            return ""  # Will be handled by _render_chart_svg method
                
        except Exception as e:
            logger.error(f"Chart rendering error: {e}")
            return ""
    
    def _render_chart_or_svg(self, chart_config: Dict[str, Any]) -> str:
        """Render chart, using SVG fallback if Plotly/Kaleido fails."""
        chart_type = chart_config.get("type", "bar")
        data = chart_config.get("data", {})
        title = chart_config.get("title", "")
        
        # Try Plotly first
        base64_img = self._render_chart_to_base64(chart_config)
        if base64_img:
            return f'<img src="data:image/png;base64,{base64_img}" alt="Chart" style="max-width:100%;" />'
        
        # SVG fallback
        if chart_type in ["bar", "barh"]:
            svg = self._render_svg_bar_chart(data, title)
            return svg if svg else '<div style="color:#999;text-align:center;">Chart unavailable</div>'
        elif chart_type == "pie":
            svg = self._render_svg_pie_chart(data, title)
            return svg if svg else '<div style="color:#999;text-align:center;">Chart unavailable</div>'
        else:
            # For other chart types, show data as table instead
            return '<div style="color:#999;text-align:center;padding:20px;">Chart type not supported in fallback mode</div>'
    
    def _generate_html(self) -> str:
        """Generate complete PowerBI-style HTML dashboard report."""
        is_vn = self.language == "vi"
        
        # PowerBI-style CSS
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
            
            :root {
                --pbi-primary: #117A65;
                --pbi-secondary: #1A5276;
                --pbi-accent: #F39C12;
                --pbi-success: #27AE60;
                --pbi-warning: #E67E22;
                --pbi-danger: #E74C3C;
                --pbi-dark: #2C3E50;
                --pbi-light: #ECF0F1;
                --pbi-bg: #F4F6F9;
                --pbi-card: #FFFFFF;
                --pbi-border: #E0E4E8;
                --pbi-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: var(--pbi-dark);
                background: var(--pbi-bg);
                padding: 0;
            }
            
            /* ===== POWERBI HEADER ===== */
            .pbi-header {
                background: linear-gradient(135deg, var(--pbi-primary) 0%, var(--pbi-secondary) 100%);
                color: white;
                padding: 24px 32px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .pbi-header-left h1 {
                font-size: 22pt;
                font-weight: 700;
                margin-bottom: 4px;
                letter-spacing: -0.5px;
            }
            
            .pbi-header-left .subtitle {
                font-size: 11pt;
                opacity: 0.9;
            }
            
            .pbi-header-right {
                text-align: right;
                font-size: 9pt;
                opacity: 0.85;
            }
            
            .pbi-header-right .logo {
                font-size: 16pt;
                font-weight: 700;
                margin-bottom: 4px;
            }
            
            /* ===== DASHBOARD CONTAINER ===== */
            .pbi-dashboard {
                padding: 20px 24px;
                max-width: 1400px;
                margin: 0 auto;
            }
            
            /* ===== KPI CARDS ROW ===== */
            .pbi-kpi-row {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-bottom: 20px;
            }
            
            .pbi-kpi-card {
                background: var(--pbi-card);
                border-radius: 8px;
                padding: 16px 20px;
                box-shadow: var(--pbi-shadow);
                border-left: 4px solid var(--pbi-primary);
                position: relative;
                overflow: hidden;
            }
            
            .pbi-kpi-card.accent { border-left-color: var(--pbi-accent); }
            .pbi-kpi-card.success { border-left-color: var(--pbi-success); }
            .pbi-kpi-card.warning { border-left-color: var(--pbi-warning); }
            .pbi-kpi-card.danger { border-left-color: var(--pbi-danger); }
            
            .pbi-kpi-card::after {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, transparent 50%, rgba(0,0,0,0.03) 50%);
            }
            
            .pbi-kpi-value {
                font-size: 28pt;
                font-weight: 700;
                color: var(--pbi-dark);
                line-height: 1.1;
            }
            
            .pbi-kpi-label {
                font-size: 9pt;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 4px;
            }
            
            .pbi-kpi-trend {
                display: inline-flex;
                align-items: center;
                font-size: 9pt;
                margin-top: 6px;
                padding: 2px 8px;
                border-radius: 12px;
            }
            
            .pbi-kpi-trend.up { background: #d4edda; color: #155724; }
            .pbi-kpi-trend.down { background: #f8d7da; color: #721c24; }
            .pbi-kpi-trend.neutral { background: #e2e3e5; color: #383d41; }
            
            /* ===== CHART GRID ===== */
            .pbi-chart-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                margin-bottom: 20px;
            }
            
            .pbi-chart-card {
                background: var(--pbi-card);
                border-radius: 8px;
                box-shadow: var(--pbi-shadow);
                overflow: hidden;
            }
            
            .pbi-chart-card.full-width {
                grid-column: 1 / -1;
            }
            
            .pbi-chart-header {
                padding: 12px 16px;
                border-bottom: 1px solid var(--pbi-border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .pbi-chart-title {
                font-size: 11pt;
                font-weight: 600;
                color: var(--pbi-dark);
            }
            
            .pbi-chart-subtitle {
                font-size: 8pt;
                color: #6c757d;
            }
            
            .pbi-chart-body {
                padding: 16px;
                text-align: center;
            }
            
            .pbi-chart-body img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }
            
            /* ===== EXECUTIVE SUMMARY ===== */
            .pbi-summary-card {
                background: var(--pbi-card);
                border-radius: 8px;
                box-shadow: var(--pbi-shadow);
                margin-bottom: 20px;
                overflow: hidden;
            }
            
            .pbi-summary-header {
                background: linear-gradient(90deg, var(--pbi-primary), var(--pbi-secondary));
                color: white;
                padding: 12px 20px;
                font-size: 12pt;
                font-weight: 600;
            }
            
            .pbi-summary-body {
                padding: 20px;
                line-height: 1.8;
                font-size: 10.5pt;
            }
            
            /* ===== DATA TABLES ===== */
            .pbi-table-card {
                background: var(--pbi-card);
                border-radius: 8px;
                box-shadow: var(--pbi-shadow);
                margin-bottom: 20px;
                overflow: hidden;
            }
            
            .pbi-table-header {
                padding: 12px 16px;
                border-bottom: 1px solid var(--pbi-border);
                font-size: 11pt;
                font-weight: 600;
                color: var(--pbi-dark);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .pbi-table-header .count {
                font-size: 9pt;
                color: #6c757d;
                font-weight: 400;
            }
            
            table.pbi-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 9pt;
            }
            
            table.pbi-table th {
                background: #f8f9fa;
                padding: 10px 12px;
                text-align: left;
                font-weight: 600;
                color: var(--pbi-dark);
                border-bottom: 2px solid var(--pbi-border);
                text-transform: uppercase;
                font-size: 8pt;
                letter-spacing: 0.5px;
            }
            
            table.pbi-table th .sort-icon {
                opacity: 0.3;
                margin-left: 4px;
            }
            
            table.pbi-table td {
                padding: 10px 12px;
                border-bottom: 1px solid #eee;
                color: #495057;
            }
            
            table.pbi-table tr:hover {
                background: #f8f9fa;
            }
            
            .pbi-status-badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 8pt;
                font-weight: 600;
            }
            
            .pbi-status-badge.success { background: #d4edda; color: #155724; }
            .pbi-status-badge.warning { background: #fff3cd; color: #856404; }
            .pbi-status-badge.danger { background: #f8d7da; color: #721c24; }
            .pbi-status-badge.info { background: #d1ecf1; color: #0c5460; }
            
            /* ===== OSINT TOOLS TIMELINE ===== */
            .pbi-timeline {
                padding: 20px;
            }
            
            .pbi-timeline-item {
                display: flex;
                margin-bottom: 16px;
                position: relative;
            }
            
            .pbi-timeline-item::before {
                content: '';
                position: absolute;
                left: 15px;
                top: 32px;
                bottom: -16px;
                width: 2px;
                background: var(--pbi-border);
            }
            
            .pbi-timeline-item:last-child::before {
                display: none;
            }
            
            .pbi-timeline-icon {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: var(--pbi-primary);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12pt;
                flex-shrink: 0;
                margin-right: 16px;
                z-index: 1;
            }
            
            .pbi-timeline-icon.success { background: var(--pbi-success); }
            .pbi-timeline-icon.warning { background: var(--pbi-warning); }
            .pbi-timeline-icon.danger { background: var(--pbi-danger); }
            
            .pbi-timeline-content {
                flex: 1;
                background: #f8f9fa;
                padding: 12px 16px;
                border-radius: 6px;
            }
            
            .pbi-timeline-title {
                font-weight: 600;
                color: var(--pbi-dark);
                margin-bottom: 4px;
            }
            
            .pbi-timeline-meta {
                font-size: 9pt;
                color: #6c757d;
            }
            
            .pbi-timeline-findings {
                margin-top: 8px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            
            .pbi-finding-tag {
                background: white;
                border: 1px solid var(--pbi-border);
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 8pt;
            }
            
            /* ===== RECOMMENDATIONS ===== */
            .pbi-recommendations {
                padding: 20px;
            }
            
            .pbi-rec-item {
                display: flex;
                align-items: flex-start;
                margin-bottom: 12px;
                padding: 12px 16px;
                background: #f8f9fa;
                border-radius: 6px;
                border-left: 3px solid var(--pbi-primary);
            }
            
            .pbi-rec-item.high {
                border-left-color: var(--pbi-danger);
                background: #fdf2f2;
            }
            
            .pbi-rec-item.medium {
                border-left-color: var(--pbi-warning);
                background: #fefce8;
            }
            
            .pbi-rec-icon {
                width: 24px;
                height: 24px;
                margin-right: 12px;
                flex-shrink: 0;
            }
            
            .pbi-rec-text {
                font-size: 10pt;
                line-height: 1.5;
            }
            
            /* ===== FOOTER ===== */
            .pbi-footer {
                background: var(--pbi-dark);
                color: rgba(255,255,255,0.7);
                padding: 16px 24px;
                text-align: center;
                font-size: 8pt;
            }
            
            .pbi-footer strong {
                color: white;
            }
            
            /* ===== PRINT STYLES ===== */
            @media print {
                body { background: white; }
                .pbi-dashboard { padding: 10px; }
                .pbi-chart-card, .pbi-summary-card, .pbi-table-card { 
                    box-shadow: none; 
                    border: 1px solid #ddd;
                    page-break-inside: avoid;
                }
                .pbi-kpi-card { box-shadow: none; border: 1px solid #ddd; }
            }
            
            /* ===== GAUGE CHART SVG ===== */
            .gauge-container {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
        </style>
        """
        
        # Header
        html = f"""
<!DOCTYPE html>
<html lang="{self.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.report_title}</title>
    {css}
</head>
<body>
    <!-- POWERBI-STYLE HEADER -->
    <div class="pbi-header">
        <div class="pbi-header-left">
            <h1>{self.report_title}</h1>
            <div class="subtitle">{"📊 Báo Cáo Phân Tích OSINT Tài Chính" if is_vn else "📊 Financial OSINT Analysis Dashboard"}</div>
        </div>
        <div class="pbi-header-right">
            <div class="logo">🐙 OpenClaw</div>
            <div>{"Ngày tạo" if is_vn else "Generated"}: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>
    </div>
    
    <div class="pbi-dashboard">
"""
        
        # ===== KPI CARDS ROW =====
        if self.kpis:
            html += """
        <!-- KPI CARDS -->
        <div class="pbi-kpi-row">
"""
            kpi_config = [
                ("osint_coverage_score", "Độ Phủ OSINT", "OSINT Coverage", "%", "accent", "up"),
                ("tools_executed", "Tools Đã Chạy", "Tools Executed", "", "", "neutral"),
                ("tools_successful", "Thành Công", "Successful", "", "success", "up"),
                ("metadata_findings_count", "Metadata", "Metadata Found", "", "", "neutral"),
                ("financial_links_found", "Links Tài Chính", "Financial Links", "", "warning", "up"),
                ("related_entities_count", "Entities Liên Quan", "Related Entities", "", "", "neutral"),
                ("info_leak_risk", "Rủi Ro Rò Rỉ", "Leak Risk", "", "danger", "neutral"),
                ("transparency_score", "Độ Minh Bạch", "Transparency", "%", "success", "up"),
            ]
            
            shown = 0
            for key, vn_label, en_label, unit, card_class, trend in kpi_config:
                if key in self.kpis and shown < 8:
                    value = self.kpis[key]
                    label = vn_label if is_vn else en_label
                    
                    # Format value
                    if isinstance(value, float):
                        display_value = f"{value:.0f}{unit}"
                    elif isinstance(value, int):
                        display_value = f"{value}{unit}"
                    else:
                        display_value = str(value)
                    
                    trend_html = ""
                    if trend == "up":
                        trend_html = '<div class="pbi-kpi-trend up">▲ Good</div>'
                    elif trend == "down":
                        trend_html = '<div class="pbi-kpi-trend down">▼ Alert</div>'
                    
                    html += f"""
            <div class="pbi-kpi-card {card_class}">
                <div class="pbi-kpi-value">{display_value}</div>
                <div class="pbi-kpi-label">{label}</div>
                {trend_html}
            </div>
"""
                    shown += 1
            
            html += """
        </div>
"""
        
        # ===== EXECUTIVE SUMMARY =====
        if self.analysis_text:
            html += f"""
        <!-- EXECUTIVE SUMMARY -->
        <div class="pbi-summary-card">
            <div class="pbi-summary-header">
                {"📋 Tổng Quan Phân Tích" if is_vn else "📋 Executive Summary"}
            </div>
            <div class="pbi-summary-body">
                {self.analysis_text}
            </div>
        </div>
"""
        
        # ===== CHARTS GRID =====
        if self.charts:
            html += """
        <!-- CHARTS GRID -->
        <div class="pbi-chart-grid">
"""
            for i, chart_config in enumerate(self.charts):
                chart_content = self._render_chart_or_svg(chart_config)
                full_width = "full-width" if i == 0 else ""
                html += f"""
            <div class="pbi-chart-card {full_width}">
                <div class="pbi-chart-header">
                    <span class="pbi-chart-title">{chart_config.get("title", "Chart")}</span>
                    <span class="pbi-chart-subtitle">{chart_config.get("description", "")}</span>
                </div>
                <div class="pbi-chart-body">
                    {chart_content}
                </div>
            </div>
"""
            html += """
        </div>
"""
        
        # ===== OSINT TOOLS TIMELINE =====
        if self.osint_data and self.osint_data.get("osint_results"):
            osint_results = self.osint_data.get("osint_results", [])
            html += f"""
        <!-- OSINT TOOLS EXECUTION -->
        <div class="pbi-table-card">
            <div class="pbi-table-header">
                {"🔍 Timeline Thực Thi OSINT Tools" if is_vn else "🔍 OSINT Tools Execution Timeline"}
                <span class="count">{len(osint_results)} {"công cụ" if is_vn else "tools"}</span>
            </div>
            <div class="pbi-timeline">
"""
            for result in osint_results:
                status_class = "success" if result.get("success") else "danger"
                icon = "✓" if result.get("success") else "✗"
                tool_name = result.get("tool", "Unknown")
                target = result.get("target", "-")
                duration = result.get("duration_ms", 0)
                
                # Extract findings
                data = result.get("data", {})
                findings_html = ""
                if data:
                    tags = []
                    if data.get("emails"):
                        tags.append(f"📧 {len(data['emails'])} emails")
                    if data.get("authors"):
                        tags.append(f"👤 {len(data['authors'])} authors")
                    if data.get("interesting_urls"):
                        tags.append(f"🔗 {len(data['interesting_urls'])} links")
                    if data.get("metadata"):
                        tags.append(f"📄 {len(data['metadata'])} metadata")
                    
                    if tags:
                        findings_html = '<div class="pbi-timeline-findings">'
                        for tag in tags:
                            findings_html += f'<span class="pbi-finding-tag">{tag}</span>'
                        findings_html += '</div>'
                
                html += f"""
                <div class="pbi-timeline-item">
                    <div class="pbi-timeline-icon {status_class}">{icon}</div>
                    <div class="pbi-timeline-content">
                        <div class="pbi-timeline-title">{tool_name}</div>
                        <div class="pbi-timeline-meta">Target: {target} | {"Thời gian" if is_vn else "Duration"}: {duration:.0f}ms</div>
                        {findings_html}
                    </div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # ===== DATA TABLES =====
        if self.tables:
            for table_config in self.tables:
                title = table_config.get("title", "Data Table")
                columns = table_config.get("columns", [])
                data = table_config.get("data", [])
                
                html += f"""
        <div class="pbi-table-card">
            <div class="pbi-table-header">
                📊 {title}
                <span class="count">{len(data)} {"dòng" if is_vn else "rows"}</span>
            </div>
            <table class="pbi-table">
                <thead>
                    <tr>
"""
                for col in columns:
                    html += f'<th>{col} <span class="sort-icon">↕</span></th>'
                
                html += """
                    </tr>
                </thead>
                <tbody>
"""
                for row in data[:50]:
                    html += "<tr>"
                    for col in columns:
                        value = row.get(col, "")
                        # Style status badges
                        if col.lower() == "status":
                            badge_class = "success" if value in ["✓", "Success", "Thành công"] else "danger"
                            value = f'<span class="pbi-status-badge {badge_class}">{value}</span>'
                        elif isinstance(value, (int, float)):
                            value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                        html += f"<td>{value}</td>"
                    html += "</tr>"
                
                html += """
                </tbody>
            </table>
        </div>
"""
        
        # ===== RECOMMENDATIONS =====
        if self.recommendations:
            html += f"""
        <!-- RECOMMENDATIONS -->
        <div class="pbi-table-card">
            <div class="pbi-table-header">
                {"💡 Khuyến Nghị Hành Động" if is_vn else "💡 Action Recommendations"}
            </div>
            <div class="pbi-recommendations">
"""
            for rec in self.recommendations:
                priority = "high" if any(w in rec.lower() for w in ["cảnh báo", "warning", "rủi ro", "risk", "ngay", "urgent"]) else \
                          "medium" if any(w in rec.lower() for w in ["nên", "should", "cần", "need"]) else ""
                icon = "⚠️" if priority == "high" else "💡" if priority == "medium" else "✅"
                
                html += f"""
                <div class="pbi-rec-item {priority}">
                    <div class="pbi-rec-icon">{icon}</div>
                    <div class="pbi-rec-text">{rec}</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # ===== FOOTER =====
        html += f"""
    </div>
    
    <!-- FOOTER -->
    <div class="pbi-footer">
        <p><strong>{"Báo cáo được tạo tự động bởi" if is_vn else "Report auto-generated by"} Leviathan OpenClaw OSINT Engine</strong></p>
        <p>{"Dữ liệu thu thập từ các nguồn OSINT công khai" if is_vn else "Data collected from public OSINT sources"} | © {datetime.now().year} Leviathan Data Intelligence Platform</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def generate_html(self, output_path: Optional[str] = None) -> str:
        """Generate HTML report and save to file."""
        html_content = self._generate_html()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_company = self.company_name.replace(" ", "_").replace("/", "_")[:30]
            output_path = str(REPORTS_DIR / f"report_{safe_company}_{timestamp}.html")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    def generate_pdf(self, output_path: Optional[str] = None) -> str:
        """Generate PDF report from HTML."""
        if not WEASYPRINT_AVAILABLE:
            logger.warning("WeasyPrint not available. Install with: pip install weasyprint")
            # Fallback: save HTML
            return self.generate_html(output_path.replace(".pdf", ".html") if output_path else None)
        
        html_content = self._generate_html()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_company = self.company_name.replace(" ", "_").replace("/", "_")[:30]
            output_path = str(REPORTS_DIR / f"report_{safe_company}_{timestamp}.pdf")
        
        try:
            html = HTML(string=html_content)
            html.write_pdf(output_path)
            logger.info(f"PDF report generated: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            # Fallback to HTML
            html_path = output_path.replace(".pdf", ".html")
            return self.generate_html(html_path)


def generate_osint_report(
    company: str,
    osint_data: Dict[str, Any],
    analysis_text: str = "",
    language: str = "vi",
    format: str = "pdf",
) -> str:
    """
    Generate a professional PowerBI-style OSINT dashboard report.
    
    Args:
        company: Company name
        osint_data: OSINT analysis results (from FinancialOSINTAnalyzer)
        analysis_text: Main analysis text from LLM
        language: Report language (vi/en)
        format: Output format (pdf/html)
    
    Returns:
        Path to generated report file
    """
    generator = ReportGenerator(
        language=language,
        company_name=company,
    )
    
    # Set data
    generator.set_kpis(osint_data.get("kpis", {}))
    generator.set_analysis(analysis_text)
    generator.set_recommendations(osint_data.get("recommendations", []))
    generator.set_osint_data(osint_data)
    
    # ===== ADD MULTIPLE CHARTS FROM OSINT DATA =====
    kpis = osint_data.get("kpis", {})
    osint_results = osint_data.get("osint_results", [])
    is_vn = language == "vi"
    
    # Chart 1: OSINT KPI Overview (Bar chart)
    if kpis:
        generator.add_chart(
            chart_type="bar",
            data={
                "x": [
                    "Coverage" if not is_vn else "Độ Phủ", 
                    "Transparency" if not is_vn else "Minh Bạch", 
                    "Success Rate" if not is_vn else "Tỷ Lệ OK"
                ],
                "y": [
                    kpis.get("osint_coverage_score", 0),
                    kpis.get("transparency_score", 0),
                    (kpis.get("tools_successful", 0) / max(kpis.get("tools_executed", 1), 1)) * 100,
                ],
            },
            title="OSINT Performance Metrics (%)" if not is_vn else "Chỉ Số Hiệu Suất OSINT (%)",
            description="Key performance indicators from OSINT data collection",
        )
    
    # Chart 2: Tools Execution Pie Chart
    if osint_results:
        success_count = sum(1 for r in osint_results if r.get("success"))
        failed_count = len(osint_results) - success_count
        
        generator.add_chart(
            chart_type="pie",
            data={
                "labels": [
                    "Success" if not is_vn else "Thành Công", 
                    "Failed" if not is_vn else "Thất Bại"
                ],
                "values": [success_count, max(failed_count, 0.1)],  # Avoid 0
            },
            title="Tool Execution Status" if not is_vn else "Trạng Thái Thực Thi Tools",
            description=f"{success_count}/{len(osint_results)} tools executed successfully",
        )
    
    # Chart 3: Findings Distribution (Bar chart)
    findings_data = {"labels": [], "values": []}
    total_emails = 0
    total_urls = 0
    total_metadata = 0
    total_authors = 0
    
    for result in osint_results:
        data = result.get("data", {})
        total_emails += len(data.get("emails", []))
        total_urls += len(data.get("interesting_urls", []))
        total_metadata += len(data.get("metadata", []))
        total_authors += len(data.get("authors", []))
    
    if any([total_emails, total_urls, total_metadata, total_authors]):
        generator.add_chart(
            chart_type="bar",
            data={
                "x": ["Emails", "URLs", "Metadata", "Authors" if not is_vn else "Tác Giả"],
                "y": [total_emails, total_urls, total_metadata, total_authors],
            },
            title="OSINT Findings by Category" if not is_vn else "Phát Hiện OSINT Theo Loại",
            description="Distribution of discovered information across categories",
        )
    
    # Chart 4: Tool Duration Comparison (Horizontal Bar)
    if osint_results:
        tool_names = [r.get("tool", "Unknown")[:15] for r in osint_results]
        tool_durations = [r.get("duration_ms", 0) for r in osint_results]
        
        generator.add_chart(
            chart_type="barh",
            data={
                "x": tool_durations,
                "y": tool_names,
            },
            title="Tool Execution Time (ms)" if not is_vn else "Thời Gian Thực Thi Tool (ms)",
            description="Comparison of execution duration across OSINT tools",
        )
    
    # ===== ADD DATA TABLES =====
    
    # Table 1: OSINT Tools Summary
    if osint_results:
        table_data = []
        for r in osint_results:
            status = "✓ OK" if r.get("success") else "✗ Fail"
            data = r.get("data", {})
            findings = []
            if data.get("emails"):
                findings.append(f"{len(data['emails'])} emails")
            if data.get("interesting_urls"):
                findings.append(f"{len(data['interesting_urls'])} URLs")
            if data.get("authors"):
                findings.append(f"{len(data['authors'])} authors")
            
            table_data.append({
                "Tool": r.get("tool", "Unknown"),
                "Target": r.get("target", "-")[:30],
                "Status": status,
                "Duration (ms)": r.get("duration_ms", 0),
                "Findings": ", ".join(findings) or "None",
            })
        
        generator.add_table(
            data=table_data,
            title="OSINT Tools Execution Details" if not is_vn else "Chi Tiết Thực Thi OSINT Tools",
            columns=["Tool", "Target", "Status", "Duration (ms)", "Findings"],
        )
    
    # Table 2: Collected Emails (if any)
    all_emails = []
    for result in osint_results:
        data = result.get("data", {})
        for email in data.get("emails", [])[:10]:
            all_emails.append({
                "Email": email,
                "Source": result.get("tool", "Unknown"),
            })
    
    if all_emails:
        generator.add_table(
            data=all_emails[:20],
            title="Discovered Email Addresses" if not is_vn else "Địa Chỉ Email Phát Hiện",
            columns=["Email", "Source"],
        )
    
    # Table 3: Financial Links (if any)
    all_links = []
    for result in osint_results:
        data = result.get("data", {})
        for url in data.get("interesting_urls", [])[:10]:
            all_links.append({
                "URL": url[:60] + "..." if len(url) > 60 else url,
                "Source": result.get("tool", "Unknown"),
            })
    
    if all_links:
        generator.add_table(
            data=all_links[:20],
            title="Financial/IR Links Discovered" if not is_vn else "Links Tài Chính/IR Phát Hiện",
            columns=["URL", "Source"],
        )
    
    # Generate report
    if format == "pdf":
        return generator.generate_pdf()
    else:
        return generator.generate_html()
