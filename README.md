# PDF_OCR

I want to analyse PDF Datasheets for electronic engineers to build better circuits with
the help of llms and AI.

I would like to use the following steps and tools:

1. Use OcrmyPdf for adding annotation to old scanned PDF files.
2. analyse the structure of PDFs using PyMuPDF, opencv2 and yolo.
3. use Spacy and easy_ocr to add additonal information with nlp.
4. analyse Schematics for subcurcuits with yolo and other tools.
5. use a MCP to feed a LLM with the generated context
6. output additonal infomation for the electronic engineer to build better PCBs

This shall help by building complex curcuits or make repair work of old PCBs and curcuits way easier.
In general it shall improve the understanding of complex curcuits.

[OcrmyPDF ](https://github.com/ocrmypdf/OCRmyPDF)

```Bash
# use this commmand if you just want to annotate your PDF file
OcrmyPdf input.pdf output.pdf --deskew --clean --rotate-pages
```

Run a local server with label studio to label data for a yolo model

```Bash
# runns a local server with label-studio
label-studio start
```

TODOs:

- [ ] label data with label-studio
- [ ] train a yolo model on the data

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

### Third-Party Dependencies

This project uses [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF), which is licensed under the [Mozilla Public License 2.0 (MPL-2.0)](https://www.mozilla.org/en-US/MPL/2.0/).

OCRmyPDF is not included in this repository, but may be installed by users as a dependency. Its license terms apply independently and do not affect the MIT license of this project.
