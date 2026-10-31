from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "GROUP_PROJECT_MID_SUBMISSION_REPORT.md"
OUTPUT = ROOT / "docs" / "GROUP_PROJECT_MID_SUBMISSION_REPORT.pdf"


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Normal"], fontName="Courier", fontSize=8, leading=10, leftIndent=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                text = "<br/>".join(clean_inline(code) for code in code_lines) or " "
                story.append(Paragraph(text, styles["CodeBlock"]))
                story.append(Spacer(1, 6))
                in_code = False
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not line.strip():
            story.append(Spacer(1, 6))
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), styles["Title"]))
            story.append(Spacer(1, 12))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), styles["Heading2"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:]), styles["Heading3"]))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            if rows:
                para_rows = [[Paragraph(clean_inline(cell), styles["Small"]) for cell in row] for row in rows]
                table = Table(para_rows, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 8))
            continue

        if line.startswith("- "):
            story.append(Paragraph("• " + clean_inline(line[2:]), styles["Normal"]))
            i += 1
            continue

        story.append(Paragraph(clean_inline(line), styles["Normal"]))
        i += 1

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)

