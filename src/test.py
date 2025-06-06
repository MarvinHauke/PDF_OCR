from pathlib import Path

import spacy
from spacy_layout import spaCyLayout

from annotator import annotate_pdf

PDF = Path("./examples/CEM33403345-VCO.pdf")
OUTPUT = Path("./output/annotated.pdf")

# Setup pipeline
nlp = spacy.load("en_core_web_sm")
layout = spaCyLayout(nlp)

# Process document
doc = layout(str(PDF))

# Annotate and save
annotate_pdf(doc, PDF, OUTPUT)
