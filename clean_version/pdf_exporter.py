#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Export System
Generate professional PDF reports from video summaries
"""

import os
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus import Image as ReportLabImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("[WARNING] ReportLab not available. Install with: pip install reportlab")

class PDFExporter:
    """Export video summaries to professional PDF reports"""
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF export. Install with: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.HexColor('#34495e'),
            borderWidth=1,
            borderColor=colors.HexColor('#bdc3c7'),
            borderPadding=10,
            backColor=colors.HexColor('#ecf0f1')
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=16,
            textColor=colors.HexColor('#e74c3c'),
            borderWidth=0,
            borderRadius=5,
            leftIndent=0
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#2c3e50'),
            leading=16
        ))
        
        # Key points style
        self.styles.add(ParagraphStyle(
            name='KeyPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leftIndent=20,
            bulletIndent=10,
            textColor=colors.HexColor('#27ae60')
        ))
        
        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_RIGHT
        ))
    
    def export_single_summary(self, summary: Dict[str, Any], output_path: str = None) -> str:
        """Export a single video summary to PDF"""
        
        if not output_path:
            safe_title = self._sanitize_filename(summary.get('title', 'summary'))
            output_path = f"exports/summary_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'exports', exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        story = []
        story.extend(self._build_header(summary))
        story.extend(self._build_summary_content(summary))
        story.extend(self._build_footer(summary))
        
        # Generate PDF
        doc.build(story)
        
        print(f"[PDF] Exported summary to: {output_path}")
        return output_path
    
    def export_batch_summary(self, summaries: List[Dict[str, Any]], output_path: str = None) -> str:
        """Export multiple summaries to a single PDF report"""
        
        if not output_path:
            output_path = f"exports/batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'exports', exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        story = []
        
        # Table of contents
        story.extend(self._build_batch_header(summaries))
        story.extend(self._build_table_of_contents(summaries))
        story.append(PageBreak())
        
        # Individual summaries
        for i, summary in enumerate(summaries):
            story.extend(self._build_summary_content(summary, section_number=i+1))
            if i < len(summaries) - 1:  # Don't add page break after last summary
                story.append(PageBreak())
        
        # Generate PDF
        doc.build(story)
        
        print(f"[PDF] Exported batch report to: {output_path}")
        return output_path
    
    def export_highlights_report(self, highlights_data: Dict[str, Any], output_path: str = None) -> str:
        """Export video highlights analysis to PDF"""
        
        if not output_path:
            safe_title = self._sanitize_filename(highlights_data.get('video_title', 'highlights'))
            output_path = f"exports/highlights_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'exports', exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        story = []
        story.extend(self._build_highlights_content(highlights_data))
        
        # Generate PDF
        doc.build(story)
        
        print(f"[PDF] Exported highlights report to: {output_path}")
        return output_path
    
    def _build_header(self, summary: Dict[str, Any]) -> List:
        """Build PDF header section"""
        content = []
        
        # Title
        title = summary.get('title', 'Video Summary')
        content.append(Paragraph(title, self.styles['CustomTitle']))
        
        # Metadata table
        metadata = summary.get('metadata', {})
        duration = metadata.get('duration', 'Unknown')
        created_at = summary.get('created_at', 'Unknown')
        ai_provider = summary.get('ai_provider', 'Unknown')
        model_used = summary.get('model_used', 'Unknown')
        
        metadata_data = [
            ['Video URL:', summary.get('url', 'N/A')],
            ['Duration:', duration],
            ['Processed:', created_at],
            ['AI Provider:', f"{ai_provider} ({model_used})"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4.5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7'))
        ]))
        
        content.append(metadata_table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _build_summary_content(self, summary: Dict[str, Any], section_number: int = None) -> List:
        """Build main summary content"""
        content = []
        
        # Section number for batch reports
        if section_number:
            content.append(Paragraph(f"Video #{section_number}", self.styles['SectionHeading']))
            content.append(Paragraph(summary.get('title', 'Untitled'), self.styles['CustomSubtitle']))
        
        # Summary section
        summary_text = summary.get('summary', 'No summary available.')
        content.append(Paragraph("📋 Summary", self.styles['SectionHeading']))
        content.append(Paragraph(summary_text, self.styles['CustomBody']))
        content.append(Spacer(1, 16))
        
        # Key points section
        key_points = summary.get('key_points', [])
        if key_points and isinstance(key_points, list) and len(key_points) > 0:
            content.append(Paragraph("🔑 Key Points", self.styles['SectionHeading']))
            
            for point in key_points:
                content.append(Paragraph(f"• {point}", self.styles['KeyPoint']))
            
            content.append(Spacer(1, 16))
        
        # Transcript section (abbreviated)
        transcript = summary.get('transcript', '')
        if transcript and len(transcript) > 100:
            content.append(Paragraph("📝 Transcript Preview", self.styles['SectionHeading']))
            
            # Show first 500 characters
            preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
            content.append(Paragraph(preview, self.styles['CustomBody']))
            content.append(Spacer(1, 16))
        
        return content
    
    def _build_batch_header(self, summaries: List[Dict[str, Any]]) -> List:
        """Build batch report header"""
        content = []
        
        content.append(Paragraph("Video Batch Analysis Report", self.styles['CustomTitle']))
        
        # Summary statistics
        total_videos = len(summaries)
        total_duration = sum([
            self._parse_duration(s.get('metadata', {}).get('duration', '0:00')) 
            for s in summaries
        ])
        
        avg_duration = total_duration / total_videos if total_videos > 0 else 0
        
        stats_data = [
            ['Total Videos:', str(total_videos)],
            ['Total Duration:', self._format_duration(total_duration)],
            ['Average Duration:', self._format_duration(avg_duration)],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2980b9'))
        ]))
        
        content.append(stats_table)
        content.append(Spacer(1, 30))
        
        return content
    
    def _build_table_of_contents(self, summaries: List[Dict[str, Any]]) -> List:
        """Build table of contents for batch report"""
        content = []
        
        content.append(Paragraph("Table of Contents", self.styles['CustomSubtitle']))
        
        toc_data = [['#', 'Video Title', 'Duration']]
        
        for i, summary in enumerate(summaries, 1):
            title = summary.get('title', 'Untitled')
            if len(title) > 60:
                title = title[:60] + "..."
            
            duration = summary.get('metadata', {}).get('duration', 'Unknown')
            toc_data.append([str(i), title, duration])
        
        toc_table = Table(toc_data, colWidths=[0.5*inch, 4.5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        content.append(toc_table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _build_highlights_content(self, highlights_data: Dict[str, Any]) -> List:
        """Build highlights report content"""
        content = []
        
        # Title
        video_title = highlights_data.get('video_title', 'Video Highlights Report')
        content.append(Paragraph(f"Highlights Analysis: {video_title}", self.styles['CustomTitle']))
        
        # Summary stats
        highlights = highlights_data.get('highlights', [])
        content.append(Paragraph("📊 Analysis Summary", self.styles['SectionHeading']))
        
        stats_text = f"""
        Total Highlights Identified: {len(highlights)}<br/>
        Analysis Method: AI-powered content analysis<br/>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        content.append(Paragraph(stats_text, self.styles['CustomBody']))
        content.append(Spacer(1, 20))
        
        # Highlights table
        if highlights:
            content.append(Paragraph("🎬 Identified Highlights", self.styles['SectionHeading']))
            
            highlights_data = [['#', 'Time', 'Duration', 'Description', 'Score']]
            
            for i, highlight in enumerate(highlights, 1):
                start_time = highlight.get('start_time', '0:00')
                duration = f"{highlight.get('duration', 60)}s"
                description = highlight.get('description', 'No description')[:50] + "..."
                score = f"{highlight.get('score', 0):.1f}"
                
                highlights_data.append([str(i), start_time, duration, description, score])
            
            highlights_table = Table(highlights_data, colWidths=[0.3*inch, 0.8*inch, 0.8*inch, 3.2*inch, 0.6*inch])
            highlights_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0392b')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fadbd8')])
            ]))
            
            content.append(highlights_table)
        
        return content
    
    def _build_footer(self, summary: Dict[str, Any]) -> List:
        """Build PDF footer"""
        content = []
        
        content.append(Spacer(1, 40))
        footer_text = f"Generated by YouTube Summarizer on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}"
        content.append(Paragraph(footer_text, self.styles['Metadata']))
        
        return content
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file creation"""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        filename = filename[:50] if len(filename) > 50 else filename
        return filename.strip()
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to seconds"""
        try:
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 2:  # MM:SS
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return int(duration_str) if duration_str.isdigit() else 0
        except:
            return 0
    
    def _format_duration(self, seconds: int) -> str:
        """Format seconds to duration string"""
        if seconds < 3600:
            return f"{seconds // 60}:{seconds % 60:02d}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}:{minutes:02d}:{seconds % 60:02d}"
    
    def get_export_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available PDF export templates"""
        return {
            'professional': {
                'name': 'Professional Report',
                'description': 'Clean, professional layout suitable for business presentations',
                'features': ['Executive summary', 'Key insights', 'Action items', 'Appendix']
            },
            'educational': {
                'name': 'Educational Summary', 
                'description': 'Detailed academic-style report with learning objectives',
                'features': ['Learning outcomes', 'Detailed analysis', 'References', 'Quiz questions']
            },
            'creative': {
                'name': 'Creative Brief',
                'description': 'Visual-focused layout for creative content analysis',
                'features': ['Visual highlights', 'Creative insights', 'Mood board', 'Inspiration notes']
            },
            'technical': {
                'name': 'Technical Documentation',
                'description': 'Structured technical report with code examples and diagrams',
                'features': ['Technical specs', 'Code snippets', 'Architecture diagrams', 'Implementation notes']
            }
        }