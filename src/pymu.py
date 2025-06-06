import pymupdf

pdf_path = "examples/CEM33403345-VCO.pdf"
doc = pymupdf.open(pdf_path)

print(doc.page_count)
print(doc.metadata)

page1 = doc.load_page(0)
page2 = doc.load_page(1)

print(page1)
print(page2.get_text())
