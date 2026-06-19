"""
make_report_pdf.py
==================
Utility that renders report.md to report.pdf using fpdf2 with an embedded
Unicode font (so the policy arrows render correctly). This is a build helper,
not part of the RL solution.

Usage:  python make_report_pdf.py
"""

import os
import re

from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "report.md")
PDF_PATH = os.path.join(HERE, "report.pdf")

FONTS = r"C:\Windows\Fonts"
SANS = os.path.join(FONTS, "arial.ttf")
SANS_B = os.path.join(FONTS, "arialbd.ttf")
MONO = os.path.join(FONTS, "consola.ttf")


def strip_inline(text):
    """Remove inline markdown emphasis markers (bold/italic/code)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


class Report(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("sans", size=8)
        self.set_text_color(140)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def build():
    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("sans", "", SANS)
    pdf.add_font("sans", "B", SANS_B)
    pdf.add_font("mono", "", MONO)
    pdf.add_page()

    with open(MD_PATH, encoding="utf-8") as f:
        lines = f.read().split("\n")

    epw = pdf.epw  # effective page width
    i = 0
    while i < len(lines):
        line = lines[i]

        # Code / pre-formatted block.
        if line.strip().startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            pdf.set_font("mono", size=9)
            pdf.set_fill_color(244, 244, 244)
            for b in block:
                pdf.cell(epw, 5, b, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            continue

        stripped = line.strip()

        if stripped == "" :
            pdf.ln(2)
        elif stripped.startswith("# "):
            pdf.set_font("sans", "B", 16)
            pdf.multi_cell(epw, 8, strip_inline(stripped[2:]))
            pdf.ln(1)
        elif stripped.startswith("### "):
            pdf.set_font("sans", "B", 11)
            pdf.multi_cell(epw, 6, strip_inline(stripped[4:]))
            pdf.ln(1)
        elif stripped.startswith("## "):
            pdf.set_font("sans", "B", 13)
            pdf.multi_cell(epw, 7, strip_inline(stripped[3:]))
            pdf.ln(1)
        elif stripped.startswith("---"):
            y = pdf.get_y()
            pdf.set_draw_color(200)
            pdf.line(pdf.l_margin, y, pdf.l_margin + epw, y)
            pdf.ln(2)
        elif stripped.startswith("|"):
            # Render markdown table rows in monospace, skipping separator rows.
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                i += 1
                continue
            pdf.set_font("mono", size=8)
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            pdf.multi_cell(epw, 5, "  ".join(cells))
        elif stripped.startswith("* ") or stripped.startswith("- "):
            pdf.set_font("sans", size=10)
            pdf.multi_cell(epw, 5, "  •  " + strip_inline(stripped[2:]))
        elif stripped.startswith(">"):
            pdf.set_font("sans", "B", 10)
            pdf.multi_cell(epw, 5, strip_inline(stripped.lstrip(">").strip()))
        else:
            pdf.set_font("sans", size=10)
            pdf.multi_cell(epw, 5, strip_inline(stripped))
        i += 1

    pdf.output(PDF_PATH)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    build()
