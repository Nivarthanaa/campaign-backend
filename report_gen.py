from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.lib import colors # type: ignore
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image # type: ignore
from reportlab.lib.units import inch # type: ignore

def make_report(path: str, title: str, sections: list):
    # Setup document
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#3b82f6"),
        alignment=1, # Center
        spaceAfter=30,
        fontName='Helvetica-Bold'
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=20,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        borderPadding=(0, 0, 5, 0),
        borderWidth=0,
        borderColor=colors.HexColor("#3b82f6"),
    )

    elements = []

    # Title
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.2 * inch))

    for section_title, items in sections:
        # Section Header
        if section_title:
            elements.append(Paragraph(section_title, section_title_style))
        
        # Handle different item types
        if isinstance(items, dict):
            if "__image__" in items:
                # Handle Image
                img_path = items["__image__"]
                img = Image(img_path, width=5.5*inch, height=3.5*inch)
                elements.append(img)
                if "__caption__" in items:
                    caption_style = ParagraphStyle(
                        'Caption',
                        parent=styles['Normal'],
                        fontSize=9,
                        textColor=colors.HexColor("#64748b"),
                        alignment=1, # Center
                        fontName='Helvetica-Oblique'
                    )
                    elements.append(Paragraph(items["__caption__"], caption_style))
            else:
                # Convert dict to Table data
                table_data = []
                for k, v in items.items():
                    table_data.append([Paragraph(f"<b>{k}</b>", styles['Normal']), Paragraph(str(v), styles['Normal'])])
                
                # Create Table
                t = Table(table_data, colWidths=[2.2 * inch, 3.8 * inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#64748b")),
                    ('ALIGN', (0, 0), (1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                    ('BOX', (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(t)
        elif isinstance(items, list):
            for line in items:
                elements.append(Paragraph(f"• {line}", styles['Normal']))
                elements.append(Spacer(1, 0.05 * inch))
        
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
