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
                        marker_color="#0088aa"
                    )
                ])
            elif chart_type == "line":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data.get("x", []),
                        y=data.get("y", []),
                        mode="lines+markers",
                        line=dict(color="#0088aa", width=2)
                    )
                ])
            elif chart_type == "pie":
                fig = go.Figure(data=[
                    go.Pie(
                        labels=data.get("labels", []),
                        values=data.get("values", []),
                        marker=dict(colors=px.colors.qualitative.Set2)
                    )
                ])
            elif chart_type == "heatmap":
                fig = go.Figure(data=[
                    go.Heatmap(
                        z=data.get("z", [[]]),
                        x=data.get("x", []),
                        y=data.get("y", []),
                        colorscale="Blues"
                    )
                ])
            elif chart_type == "scatter":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data.get("x", []),
                        y=data.get("y", []),
                        mode="markers",
                        marker=dict(color="#0088aa", size=10)
                    )
                ])
            else:
                # Default to bar
                fig = go.Figure(data=[
                    go.Bar(x=data.get("x", []), y=data.get("y", []))
                ])
            
            fig.update_layout(
                title=title,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Arial", size=12, color="black"),
                margin=dict(l=50, r=50, t=50, b=50),
            )
            
            # Convert to base64 image
            if KALEIDO_AVAILABLE:
                img_bytes = fig.to_image(format="png", width=800, height=400)
                return base64.b64encode(img_bytes).decode("utf-8")
            else:
                # Fallback: return empty (chart will be skipped)
                logger.warning("Kaleido not available for chart rendering")
                return ""
                
        except Exception as e:
            logger.error(f"Chart rendering error: {e}")
            return ""
    
    def _generate_html(self) -> str:
        """Generate complete HTML report."""
        is_vn = self.language == "vi"
        
        # CSS Styles
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #1a1a2e;
                background: #ffffff;
                padding: 40px;
            }
            
            .report-header {
                text-align: center;
                border-bottom: 3px solid #0088aa;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            
            .report-header h1 {
                font-size: 24pt;
                font-weight: 700;
                color: #0088aa;
                margin-bottom: 8px;
            }
            
            .report-header .subtitle {
                font-size: 12pt;
                color: #666;
            }
            
            .report-header .timestamp {
                font-size: 10pt;
                color: #999;
                margin-top: 8px;
            }
            
            .section {
                margin-bottom: 30px;
                page-break-inside: avoid;
            }
            
            .section-title {
                font-size: 14pt;
                font-weight: 600;
                color: #0088aa;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 8px;
                margin-bottom: 15px;
            }
            
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .kpi-card {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            
            .kpi-value {
                font-size: 24pt;
                font-weight: 700;
                color: #0088aa;
            }
            
            .kpi-label {
                font-size: 9pt;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .kpi-change {
                font-size: 10pt;
                margin-top: 5px;
            }
            
            .kpi-change.positive { color: #28a745; }
            .kpi-change.negative { color: #dc3545; }
            
            .analysis-text {
                background: #f8f9fa;
                border-left: 4px solid #0088aa;
                padding: 15px 20px;
                margin-bottom: 20px;
                font-size: 11pt;
                line-height: 1.8;
            }
            
            .recommendations-list {
                list-style: none;
                padding: 0;
            }
            
            .recommendations-list li {
                padding: 10px 15px;
                margin-bottom: 8px;
                background: #e7f5ff;
                border-radius: 6px;
                border-left: 4px solid #0088aa;
            }
            
            .recommendations-list li.warning {
                background: #fff3cd;
                border-left-color: #ffc107;
            }
            
            .chart-container {
                text-align: center;
                margin: 20px 0;
                page-break-inside: avoid;
            }
            
            .chart-container img {
                max-width: 100%;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            
            .chart-title {
                font-size: 12pt;
                font-weight: 600;
                color: #333;
                margin-bottom: 10px;
            }
            
            .chart-description {
                font-size: 10pt;
                color: #666;
                margin-top: 8px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 10pt;
            }
            
            th, td {
                padding: 10px 12px;
                text-align: left;
                border: 1px solid #dee2e6;
            }
            
            th {
                background: #0088aa;
                color: white;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 9pt;
                letter-spacing: 0.5px;
            }
            
            tr:nth-child(even) {
                background: #f8f9fa;
            }
            
            tr:hover {
                background: #e9ecef;
            }
            
            .osint-tool-result {
                margin-bottom: 15px;
                padding: 12px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
            
            .osint-tool-name {
                font-weight: 600;
                color: #0088aa;
                font-size: 11pt;
            }
            
            .osint-tool-status {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: 500;
            }
            
            .osint-tool-status.success {
                background: #d4edda;
                color: #155724;
            }
            
            .osint-tool-status.failed {
                background: #f8d7da;
                color: #721c24;
            }
            
            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                text-align: center;
                font-size: 9pt;
                color: #999;
            }
            
            @media print {
                body { padding: 20px; }
                .section { page-break-inside: avoid; }
                .chart-container { page-break-inside: avoid; }
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
    <div class="report-header">
        <h1>{self.report_title}</h1>
        <div class="subtitle">{"Báo cáo phân tích dữ liệu OSINT tài chính" if is_vn else "Financial OSINT Data Analysis Report"}</div>
        <div class="timestamp">{"Ngày tạo" if is_vn else "Generated"}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
"""
        
        # Executive Summary / KPIs
        if self.kpis:
            html += f"""
    <div class="section">
        <h2 class="section-title">{"Tổng Quan KPI" if is_vn else "KPI Overview"}</h2>
        <div class="kpi-grid">
"""
            kpi_labels = {
                "osint_coverage_score": ("Độ phủ OSINT", "OSINT Coverage") ,
                "tools_executed": ("Công cụ đã chạy", "Tools Executed"),
                "tools_successful": ("Thành công", "Successful"),
                "metadata_findings_count": ("Phát hiện Metadata", "Metadata Findings"),
                "financial_links_found": ("Links Tài chính", "Financial Links"),
                "info_leak_risk": ("Rủi ro rò rỉ", "Leak Risk"),
                "transparency_score": ("Độ minh bạch", "Transparency"),
            }
            
            for key, value in list(self.kpis.items())[:8]:  # Max 8 KPIs
                label = kpi_labels.get(key, (key, key))
                label_text = label[0] if is_vn else label[1]
                
                # Format value
                if isinstance(value, float):
                    display_value = f"{value:.1f}"
                elif isinstance(value, int):
                    display_value = str(value)
                else:
                    display_value = str(value)
                
                # Add unit for scores
                if "score" in key.lower():
                    display_value += "%"
                
                html += f"""
            <div class="kpi-card">
                <div class="kpi-value">{display_value}</div>
                <div class="kpi-label">{label_text}</div>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        # Analysis Text
        if self.analysis_text:
            html += f"""
    <div class="section">
        <h2 class="section-title">{"Phân Tích Chi Tiết" if is_vn else "Detailed Analysis"}</h2>
        <div class="analysis-text">
            {self.analysis_text}
        </div>
    </div>
"""
        
        # OSINT Tool Results
        if self.osint_data and self.osint_data.get("osint_results"):
            html += f"""
    <div class="section">
        <h2 class="section-title">{"Kết Quả OSINT Tools" if is_vn else "OSINT Tool Results"}</h2>
"""
            for result in self.osint_data.get("osint_results", []):
                status_class = "success" if result.get("success") else "failed"
                status_text = ("Thành công" if is_vn else "Success") if result.get("success") else ("Thất bại" if is_vn else "Failed")
                
                html += f"""
        <div class="osint-tool-result">
            <span class="osint-tool-name">{result.get("tool", "Unknown")}</span>
            <span class="osint-tool-status {status_class}">{status_text}</span>
            <span style="margin-left: 10px; color: #666; font-size: 10pt;">
                Target: {result.get("target", "-")} | 
                Duration: {result.get("duration_ms", 0):.0f}ms
            </span>
"""
                # Show key findings
                data = result.get("data", {})
                if data:
                    findings = []
                    if data.get("authors"):
                        findings.append(f"Authors: {', '.join(data['authors'][:3])}")
                    if data.get("emails"):
                        findings.append(f"Emails: {len(data['emails'])} found")
                    if data.get("interesting_urls"):
                        findings.append(f"IR Links: {len(data['interesting_urls'])}")
                    
                    if findings:
                        html += f"""
            <div style="margin-top: 8px; font-size: 10pt; color: #666;">
                {"Phát hiện: " if is_vn else "Findings: "}{" | ".join(findings)}
            </div>
"""
                
                html += """
        </div>
"""
            html += """
    </div>
"""
        
        # Charts
        if self.charts:
            html += f"""
    <div class="section">
        <h2 class="section-title">{"Biểu Đồ Phân Tích" if is_vn else "Analysis Charts"}</h2>
"""
            for chart_config in self.charts:
                chart_img = self._render_chart_to_base64(chart_config)
                if chart_img:
                    html += f"""
        <div class="chart-container">
            <div class="chart-title">{chart_config.get("title", "")}</div>
            <img src="data:image/png;base64,{chart_img}" alt="Chart" />
            <div class="chart-description">{chart_config.get("description", "")}</div>
        </div>
"""
            html += """
    </div>
"""
        
        # Data Tables
        if self.tables:
            html += f"""
    <div class="section">
        <h2 class="section-title">{"Bảng Dữ Liệu" if is_vn else "Data Tables"}</h2>
"""
            for table_config in self.tables:
                title = table_config.get("title", "")
                columns = table_config.get("columns", [])
                data = table_config.get("data", [])
                
                if title:
                    html += f'<h3 style="font-size: 12pt; margin: 15px 0 10px;">{title}</h3>'
                
                html += "<table><thead><tr>"
                for col in columns:
                    html += f"<th>{col}</th>"
                html += "</tr></thead><tbody>"
                
                for row in data[:50]:  # Max 50 rows
                    html += "<tr>"
                    for col in columns:
                        value = row.get(col, "")
                        if isinstance(value, (int, float)):
                            value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                        html += f"<td>{value}</td>"
                    html += "</tr>"
                
                html += "</tbody></table>"
            
            html += """
    </div>
"""
        
        # Recommendations
        if self.recommendations:
            html += f"""
    <div class="section">
        <h2 class="section-title">{"Khuyến Nghị" if is_vn else "Recommendations"}</h2>
        <ul class="recommendations-list">
"""
            for rec in self.recommendations:
                css_class = "warning" if any(w in rec.lower() for w in ["cảnh báo", "warning", "rủi ro", "risk"]) else ""
                html += f'<li class="{css_class}">{rec}</li>'
            
            html += """
        </ul>
    </div>
"""
        
        # Footer
        html += f"""
    <div class="footer">
        <p>{"Báo cáo được tạo bởi Leviathan OpenClaw OSINT Engine" if is_vn else "Report generated by Leviathan OpenClaw OSINT Engine"}</p>
        <p>© {datetime.now().year} Leviathan Data Intelligence Platform</p>
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
    Generate a professional OSINT report.
    
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
    
    # Add charts from OSINT data
    kpis = osint_data.get("kpis", {})
    
    # Coverage chart
    if kpis:
        generator.add_chart(
            chart_type="bar",
            data={
                "x": ["Coverage", "Transparency", "Tools OK"],
                "y": [
                    kpis.get("osint_coverage_score", 0),
                    kpis.get("transparency_score", 0),
                    (kpis.get("tools_successful", 0) / max(kpis.get("tools_executed", 1), 1)) * 100,
                ],
            },
            title="OSINT Metrics (%)" if language == "en" else "Chỉ số OSINT (%)",
            description="Key performance indicators from OSINT analysis",
        )
    
    # Add tables
    osint_results = osint_data.get("osint_results", [])
    if osint_results:
        table_data = [
            {
                "Tool": r.get("tool", ""),
                "Target": r.get("target", ""),
                "Status": "✓" if r.get("success") else "✗",
                "Duration (ms)": r.get("duration_ms", 0),
            }
            for r in osint_results
        ]
        generator.add_table(
            data=table_data,
            title="OSINT Tools Execution Summary" if language == "en" else "Tóm tắt thực thi OSINT Tools",
            columns=["Tool", "Target", "Status", "Duration (ms)"],
        )
    
    # Generate report
    if format == "pdf":
        return generator.generate_pdf()
    else:
        return generator.generate_html()
