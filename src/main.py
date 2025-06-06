# remove annoying warning
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

import spacy
from spacy_layout import spaCyLayout

PDF = "./examples/CEM33403345-VCO.pdf"

nlp = spacy.load("en_core_web_sm")
layout = spaCyLayout(nlp)
doc = layout(PDF)
# print(doc.text)
# print(doc.spans)
# print(doc._.markdown)
# print(doc._.tables)
print(doc._.layout)


# for section in doc._.pages[2][1]:
#     print("label:", section.label_)
#     print("text:", section.text)
#     print("layout:", section._.layout)

import matplotlib.pyplot as plt
import numpy as np
import pypdfium2 as pdfium
from matplotlib.patches import Rectangle

# Load and convert the PDF page to an image
page_index = 2
pdf = pdfium.PdfDocument(PDF)
page_image = pdf[page_index].render(scale=1)  # Get page 3 (index 2)
numpy_array = page_image.to_numpy()

# Get page 3 layout and sections
page = doc._.pages[page_index]
page_layout = doc._.layout.pages[page_index]

# Get scale between PDF layout size and rendered image
layout_width = page_layout.width
layout_height = page_layout.height
image_height, image_width, _ = numpy_array.shape

x_scale = image_width / layout_width
y_scale = image_height / layout_height

# Get figure and axis with page dimensions
fig, ax = plt.subplots(figsize=[12, 16])

# display the PDF image
ax.imshow(numpy_array)

# Add rectangles for each section's bounding box
for section in page[1]:
    layout = section._.layout
    # Create rectangle patch
    rect = Rectangle(
        (layout.x, layout.y),
        layout.width,
        layout.height,
        fill=False,
        color="blue",
        linewidth=1,
        alpha=0.5,
    )
    ax.add_patch(rect)

    # Add text label at top of box
    ax.text(
        layout.x,
        layout.y,
        section.label_,
        fontsize=8,
        color="red",
        verticalalignment="bottom",
    )
    # Set title and display
    ax.set_title("Page 3 layout with bounding Boxes")
    ax.axis("off")  # Hide axes
    plt.show()
