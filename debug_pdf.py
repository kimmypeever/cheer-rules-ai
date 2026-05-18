"""
Diagnose what pdfplumber can extract from the rulebook tables.
Run: python debug_pdf.py
"""
import pdfplumber
from pathlib import Path

PDF = next(Path("data/rulebooks").glob("*.pdf"))

with pdfplumber.open(PDF) as pdf:
    for page_num in [15, 19]:
        page = pdf.pages[page_num - 1]
        tables = page.extract_tables({"text_tolerance": 5})

        print(f"=== PAGE {page_num} ===")
        print(f"Tables found: {len(tables)}")

        if tables:
            t = tables[0]
            print(f"Rows: {len(t)}  Cols: {len(t[0]) if t else 0}")
            for i, row in enumerate(t[:6]):
                print(f"  Row {i}: {[str(c or '')[:40] for c in row]}")
        else:
            print("No tables — layout text:")
            print((page.extract_text(layout=True) or "")[:400])

        print()
