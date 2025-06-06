# PDF_OCR

I want to analyse PDF Datasheets with the following steps:

> [!IMPORTANT]

1. Use OcrmyPdf for adding text annotation to scanned PDF file.
2. analyse the structure of a PDF using PyMuPDF
3. use Spacy, easy_ocr to add additonal information
4. use a MCP to feed a LLM with context for building PCBs

This shall help by building nex curcuits or reparing old PCBs.
In general it shall improve the understanding of complex curcuits.

if you just want to convert your PDF run this

```Bash
# This is
OcrmyPdf input.pdf output.pdf
```
