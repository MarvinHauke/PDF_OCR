import warnings
from pathlib import Path

# suppress warning from pytorch
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

import spacy
from spacy_layout import spaCyLayout

PDF = Path("./examples/CEM33403345-VCO.pdf")
OUTPUT = Path("./output/annotated.pdf")

# Setup pipeline
# _sm stands for small model for more accurate results use _md or _lg
nlp = spacy.load("en_core_web_md")

# Add sentencizer
nlp.add_pipe("sentencizer")

layout = spaCyLayout(nlp)

# Process document .... this will take a while
doc = layout(str(PDF))
sentences = list(doc.sents)
print(doc)

print(doc._.layout)
print(doc.sentences[1])
