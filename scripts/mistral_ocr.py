import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from mistralai import DocumentURLChunk, Mistral
from mistralai.extra import response_format_from_pydantic_model
from mistralai.models import OCRResponse
from pydantic import BaseModel, Field
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

load_dotenv()


app = typer.Typer(name="mistral-ocr", help="Mistral OCR API")


api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    logger.error("MISTRAL_API_KEY not found in environment variables")
    raise ValueError("MISTRAL_API_KEY is required")

client = Mistral(api_key=api_key)
logger.info("Mistral client initialized successfully")


class ImageAnnotation(BaseModel):
    image_type: str = Field(description="The type of image")
    short_description: str = Field(description="A short description of the image")
    summary: str = Field(description="A summary of the image")


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    logger.debug(f"Replacing {len(images_dict)} images in markdown")
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    logger.debug("Image replacement completed")
    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse) -> str:
    logger.info(f"Combining markdown from {len(ocr_response.pages)} pages")
    markdowns: list[str] = []
    for i, page in enumerate(ocr_response.pages):
        logger.debug(f"Processing page {i + 1}")
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    combined_markdown = "\n\n".join(markdowns)
    logger.info(f"Combined markdown created with {len(combined_markdown)} characters")
    return combined_markdown


def ocr_pdf_file_to_markdown(file_path: str, output_path: str):
    logger.info(f"Starting OCR process for file: {file_path}")

    try:
        # Read PDF
        pdf_file = Path(file_path)
        if not pdf_file.is_file():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(
            f"Reading PDF file: {pdf_file.name} ({pdf_file.stat().st_size} bytes)"
        )

        # Upload PDF
        logger.info("Uploading PDF to Mistral API")
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_file.stem,
                "content": pdf_file.read_bytes(),
            },
            purpose="ocr",
        )
        logger.info(f"File uploaded successfully with ID: {uploaded_file.id}")

        # Get signed URL
        logger.info("Getting signed URL for uploaded file")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        logger.debug(f"Signed URL obtained: {signed_url.url[:50]}...")

        # OCR
        logger.info("Starting OCR processing")
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True,
        )
        logger.info("OCR processing completed successfully")

        # Output to Markdown
        logger.info("Converting OCR response to markdown")
        output_markdown = get_combined_markdown(pdf_response)

        logger.info(f"Writing output to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_markdown)

        logger.info(
            f"OCR process completed successfully. Output saved to: {output_path}"
        )

    except Exception as e:
        logger.error(f"OCR process failed: {str(e)}", exc_info=True)
        raise


def ocr_pdf_object(
    pdf_path: Optional[str] = None,
    file_id: Optional[str] = None,
    image_annotation_model: Optional[BaseModel] = None,
) -> OCRResponse:
    """
    Process a PDF file using Mistral OCR API.

    Args:
        pdf_path: Path to a local PDF file to upload and process
        file_id: ID of a previously uploaded file to process
        image_annotation_model: Pydantic model to annotate images
    """
    # Comprehensive input validation
    if not pdf_path and not file_id:
        logger.error("Either pdf_path or file_id must be provided")
        raise ValueError("Either pdf_path or file_id must be provided")

    if pdf_path and file_id:
        logger.warning(
            "Both pdf_path and file_id provided. Using pdf_path and ignoring file_id. "
            "Consider providing only one parameter for clarity."
        )

    try:
        if pdf_path:
            # Validate file path
            pdf_file = Path(pdf_path)
            if not pdf_file.is_file():
                logger.error(f"File not found: {pdf_path}")
                raise FileNotFoundError(f"File not found: {pdf_path}")

            # Basic PDF validation (check extension)
            if pdf_file.suffix.lower() != ".pdf":
                logger.warning(
                    f"File {pdf_path} doesn't have .pdf extension, proceeding anyway"
                )

            logger.info(f"Uploading PDF file: {pdf_file.name}")
            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )
            logger.info(f"File uploaded successfully with ID: {uploaded_file.id}")

            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            logger.debug(
                f"Signed URL obtained for uploaded file: {signed_url.url[:50]}..."
            )
        else:
            # Using existing file_id
            logger.info(f"Using existing file with ID: {file_id}")
            signed_url = client.files.get_signed_url(file_id=file_id, expiry=1)
            logger.debug(f"Signed URL obtained for file_id: {signed_url.url[:50]}...")

        logger.info("Starting OCR processing")
        if image_annotation_model:
            bbox_annotation_format = response_format_from_pydantic_model(
                image_annotation_model
            )
        else:
            bbox_annotation_format = None

        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True,
            bbox_annotation_format=bbox_annotation_format,
        )
        logger.info("OCR processing completed successfully")
        return pdf_response

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise

    except Exception as e:
        logger.error(f"OCR process failed: {str(e)}", exc_info=True)
        raise


@app.command(name="run")
def run(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to process"),
    output_path: str = typer.Argument(..., help="Path to the output file"),
    file_id: str = typer.Argument(
        default=None,
        help="ID of the file to process, if not provided, pdf_path is used",
    ),
    persist_json: bool = typer.Option(
        False,
        "--persist-json",
        help="Whether to persist the JSON response to a file",
    ),
):
    result: OCRResponse = ocr_pdf_object(pdf_path, file_id, ImageAnnotation)

    markdown_path = Path(output_path).with_suffix(".md")
    output_markdown = get_combined_markdown(result)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(output_markdown)
    logger.info(f"Markdown file persisted to {markdown_path}")

    if persist_json:
        json_path = Path(output_path).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=4)
        logger.info(f"JSON response persisted to {json_path}")


if __name__ == "__main__":
    app()
