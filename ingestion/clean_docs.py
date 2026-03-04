import os
from docx import Document
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "policies"
OUTPUT_DIR = BASE_DIR / "cleaned_texts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text(docx_path):
    doc = Document(docx_path)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".docx"):
        full_path = os.path.join(INPUT_DIR, file)
        text = extract_text(full_path)

        output_file = file.replace(".docx", ".txt")
        with open(os.path.join(OUTPUT_DIR, output_file), "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Extracted: {file}")
