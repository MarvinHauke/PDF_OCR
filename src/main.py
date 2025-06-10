import os

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
PDF = "./input/CEM33403345-VCO.pdf"
result = converter.convert(PDF)
document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()
print(markdown_output)
os.makedirs("./tmp/markdown/", exist_ok=True)

with open("./tmp/markdown/file.md", "w", encoding="utf-8") as f:
    f.write(markdown_output)
