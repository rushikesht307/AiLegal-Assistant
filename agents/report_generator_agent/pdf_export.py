import os
import re
from datetime import datetime

REPORT_DIR = os.path.join("storage", "generated_reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def _clean_md(text):
    """Remove markdown symbols so the PDF is clean text."""
    text = str(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **bold**  -> bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)        # *italic*  -> italic
    text = re.sub(r"_(.+?)_", r"\1", text)          # _note_    -> note
    text = re.sub(r"^\s*\*\s+", "- ", text)         # "* item"  -> "- item"
    text = text.replace("**", "").replace("__", "")
    return text


def export_report_pdf(sections: dict, filename: str = "legal_report") -> str:
    """sections = { "Section Title": "text", ... }  ->  writes a clean PDF, returns the path."""
    pdf_path = os.path.join(REPORT_DIR, f"{filename}.pdf")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        styles = getSampleStyleSheet()
        navy = colors.HexColor("#0F2B46")
        green = colors.HexColor("#10a37f")
        title_style = ParagraphStyle("t", parent=styles["Title"], fontSize=20,
                                     textColor=navy, spaceAfter=4)
        date_style = ParagraphStyle("d", parent=styles["Normal"], fontSize=10,
                                    textColor=colors.HexColor("#666666"), spaceAfter=10)
        head_style = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13,
                                    textColor=green, spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle("b", parent=styles["Normal"], fontSize=10,
                                    leading=14.5, spaceAfter=3)

        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                leftMargin=20*mm, rightMargin=20*mm,
                                topMargin=20*mm, bottomMargin=18*mm)
        story = [
            Paragraph("AI Legal Assistant &ndash; Analysis Report", title_style),
            Paragraph(datetime.now().strftime("Generated on %d %b %Y, %H:%M"), date_style),
            HRFlowable(width="100%", thickness=1, color=navy),
            Spacer(1, 10),
        ]
        for i, (title, text) in enumerate(sections.items(), start=1):
            story.append(Paragraph(f"{i}. {title}", head_style))
            for line in str(text).split("\n"):
                line = _clean_md(line)
                if line.strip():
                    story.append(Paragraph(line.replace("&", "&amp;"), body_style))
            story.append(Spacer(1, 8))
        doc.build(story)
        return pdf_path

    except Exception:
        # fallback: plain text (still cleaned)
        txt_path = os.path.join(REPORT_DIR, f"{filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("AI LEGAL ASSISTANT - ANALYSIS REPORT\n\n")
            for i, (title, text) in enumerate(sections.items(), start=1):
                f.write(f"{i}. {title}\n{_clean_md(text)}\n\n")
        return txt_path

 