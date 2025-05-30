from pathlib import Path

from mistralai import DocumentURLChunk, Mistral
from mistralai.models import OCRResponse

api_key = "API_KEY"
client = Mistral(api_key=api_key)


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)


def ocr_pdf_file(file_path: str, output_path: str):
    # Read PDF
    pdf_file = Path(file_path)
    assert pdf_file.is_file()

    # Upload PDF
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    # OCR
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )

    # Output to Markdown
    output_markdown = get_combined_markdown(pdf_response)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_markdown)


ocr_pdf_file("document.pdf", "output.md")
