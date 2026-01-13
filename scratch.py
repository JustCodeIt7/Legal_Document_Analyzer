#%%
from langchain_docling.loader import DoclingLoader

import markitdown
FILE_PATH = "./Lease_Agreement.pdf"
#%%
loader = DoclingLoader(file_path=FILE_PATH)

docs = loader.load()

print(f"Number of pages: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- Page {i + 1} ---\n")
    print(doc.page_content)
    
#%%    


src_file_path: str = FILE_PATH

md = markitdown.MarkItDown()
result = md.convert(src_file_path)
print(result.markdown)
with open("markitdown-poc-output.md", "w", encoding="utf-8") as f:
    f.write(result.markdown)
    
print(result)
# %%
import pymupdf4llm

def convert_pdf_to_md(pdf_path, output_path):
    # Convert PDF pages to a list of Markdown strings
    md_text_list = pymupdf4llm.to_markdown(pdf_path)
    


    # Save to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text_list)



# %%
convert_pdf_to_md(FILE_PATH, "output.md")
# %%
import fitz  # PyMuPDF for PDFs
import ollama
import io
from PIL import Image

def convert_pdf_to_images(pdf_path):
    images = []
    doc = fitz.open(pdf_path)  # Open the PDF
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap()  # Render page to pixel map
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL image
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")  # Save as in-memory PNG
        images.append(img_buffer.getvalue())  # Raw PNG bytes
    return images

prompt = "Extract all readable text from these images and format it as structured Markdown."
def query_llm_with_images(image_bytes_list, model="gemma3:latest", prompt=prompt):
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": image_bytes_list
        }]
    )
    return response["message"]["content"]

if __name__ == '__main__':

    pdf_path = "mypdf.pdf"  # Replace with your PDF file
    images = convert_pdf_to_images(pdf_path)

    if images:
        print(f"Converted {len(images)} pages to images.")
    
        extracted_text = query_llm_with_images(images)
    
        with open("output.md", "w", encoding="utf-8") as md_file:
            md_file.write(extracted_text)
        print("\nMarkdown Conversion Complete! Check `output.md`.")
    else:
        print("No images found in the PDF.")

# %%
import pymupdf
from pymupdf_rag import to_markdown  # import Markdown converter

doc = pymupdf.open(FILE_PATH)  # open input PDF

# define desired pages: this corresponds “-pages 1-10,15,20-N”
page_list = list(range(9)) + [14] + list(range(19, len(doc) - 1))

# get markdown string for all pages
md_text = to_markdown(doc, pages=page_list)

# write markdown string to some file
output = open("out-markdown.md", "w", encoding="utf-8")
output.write(md_text)
output.close()

# %%
# convert the document to markdown
import pymupdf4llm
md_text = pymupdf4llm.to_markdown(FILE_PATH)

# Write the text to some file in UTF8-encoding
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
# %%
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter

# Get the MD text
md_text = pymupdf4llm.to_markdown("input.pdf")  # get markdown for all pages

splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

splitter.create_documents([md_text])