# PDF_OCR

I want to analyse PDF Datasheets with the following steps:

1. Use OcrmyPdf for adding text annotation to scanned PDF file.
2. analyse the structure of a PDF using PyMuPDF, PDF
3. use Spacy, easy_ocr to add additonal information
4. use a MCP to feed a LLM with context for building PCBs

This shall help by building nex curcuits or reparing old PCBs.
In general it shall improve the understanding of complex curcuits.

[OcrmyPDF ](https://github.com/ocrmypdf/OCRmyPDF)

```Bash
#use this commmand if you just want to annotate your PDF file
OcrmyPdf input.pdf output.pdf
```
