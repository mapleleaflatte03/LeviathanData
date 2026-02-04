"""Tool: mm_tesseract
Tesseract OCR for text extraction from images.

Supported operations:
- ocr: Extract text from image
- ocr_data: Get detailed OCR data with bounding boxes
- detect_orientation: Detect text orientation
- get_languages: List available languages
- pdf_to_text: Extract text from PDF
"""
from typing import Any, Dict, List, Optional
import json
import os
import subprocess
import shutil


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


pytesseract = _optional_import("pytesseract")
PIL = _optional_import("PIL")
pdf2image = _optional_import("pdf2image")


def _check_tesseract() -> bool:
    """Check if tesseract is installed."""
    return shutil.which("tesseract") is not None


def _get_image(image_path: str) -> Any:
    """Load image from path."""
    if PIL is None:
        raise ImportError("Pillow is not installed. Run: pip install Pillow")
    
    from PIL import Image
    return Image.open(image_path)


def _ocr(
    image_path: str,
    lang: str = "eng",
    config: str = "",
    psm: Optional[int] = None,
    oem: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract text from image."""
    if pytesseract is None:
        raise ImportError("pytesseract is not installed. Run: pip install pytesseract")
    
    if not _check_tesseract():
        raise RuntimeError("Tesseract is not installed. Install with: apt install tesseract-ocr")
    
    image = _get_image(image_path)
    
    # Build config string
    config_parts = [config] if config else []
    if psm is not None:
        config_parts.append(f"--psm {psm}")
    if oem is not None:
        config_parts.append(f"--oem {oem}")
    
    final_config = " ".join(config_parts)
    
    text = pytesseract.image_to_string(image, lang=lang, config=final_config)
    
    return {
        "text": text.strip(),
        "language": lang,
        "image_size": image.size,
    }


def _ocr_data(
    image_path: str,
    lang: str = "eng",
    output_type: str = "dict",
) -> Dict[str, Any]:
    """Get detailed OCR data with bounding boxes."""
    if pytesseract is None:
        raise ImportError("pytesseract is not installed")
    
    if not _check_tesseract():
        raise RuntimeError("Tesseract is not installed")
    
    image = _get_image(image_path)
    
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    
    # Process into word-level results
    words = []
    for i in range(len(data["text"])):
        if data["text"][i].strip():
            words.append({
                "text": data["text"][i],
                "confidence": data["conf"][i],
                "bbox": {
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                },
                "block_num": data["block_num"][i],
                "line_num": data["line_num"][i],
                "word_num": data["word_num"][i],
            })
    
    # Group by lines
    lines = {}
    for word in words:
        key = (word["block_num"], word["line_num"])
        if key not in lines:
            lines[key] = []
        lines[key].append(word)
    
    line_texts = [" ".join(w["text"] for w in line_words) for line_words in lines.values()]
    
    return {
        "words": words,
        "lines": line_texts,
        "word_count": len(words),
        "line_count": len(lines),
    }


def _detect_orientation(image_path: str) -> Dict[str, Any]:
    """Detect text orientation and script."""
    if pytesseract is None:
        raise ImportError("pytesseract is not installed")
    
    if not _check_tesseract():
        raise RuntimeError("Tesseract is not installed")
    
    image = _get_image(image_path)
    
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return {
            "orientation": osd.get("orientation", 0),
            "rotate": osd.get("rotate", 0),
            "script": osd.get("script", "Unknown"),
            "script_confidence": osd.get("script_conf", 0),
            "orientation_confidence": osd.get("orientation_conf", 0),
        }
    except Exception as e:
        return {
            "error": str(e),
            "orientation": 0,
            "script": "Unknown",
        }


def _get_languages() -> Dict[str, Any]:
    """Get list of available Tesseract languages."""
    if not _check_tesseract():
        raise RuntimeError("Tesseract is not installed")
    
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split("\n")
        # First line is usually path info, skip it
        languages = [l.strip() for l in lines[1:] if l.strip()]
        return {"languages": languages}
    except Exception as e:
        return {"error": str(e), "languages": []}


def _pdf_to_text(
    pdf_path: str,
    lang: str = "eng",
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract text from PDF using OCR."""
    if pdf2image is None:
        raise ImportError("pdf2image is not installed. Run: pip install pdf2image")
    
    if pytesseract is None:
        raise ImportError("pytesseract is not installed")
    
    from pdf2image import convert_from_path
    
    kwargs = {"dpi": dpi}
    if first_page:
        kwargs["first_page"] = first_page
    if last_page:
        kwargs["last_page"] = last_page
    
    images = convert_from_path(pdf_path, **kwargs)
    
    pages = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang=lang)
        pages.append({
            "page": i + 1,
            "text": text.strip(),
        })
    
    full_text = "\n\n".join(p["text"] for p in pages)
    
    return {
        "pages": pages,
        "page_count": len(pages),
        "full_text": full_text,
    }


def _ocr_to_searchable_pdf(
    image_path: str,
    output_path: str,
    lang: str = "eng",
) -> Dict[str, Any]:
    """Create searchable PDF from image."""
    if pytesseract is None:
        raise ImportError("pytesseract is not installed")
    
    if not _check_tesseract():
        raise RuntimeError("Tesseract is not installed")
    
    image = _get_image(image_path)
    
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, lang=lang, extension="pdf")
    
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    
    return {
        "output_path": output_path,
        "size": len(pdf_bytes),
    }


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Tesseract OCR operations.
    
    Args:
        args: Dictionary with:
            - operation: "ocr", "ocr_data", "detect_orientation", "languages", "pdf_to_text", "to_pdf"
            - image_path: Path to input image
            - lang: Language code (default: "eng")
            - Various operation-specific parameters
    
    Returns:
        Result dictionary with extracted text and metadata
    """
    args = args or {}
    operation = args.get("operation", "ocr")
    
    if not _check_tesseract() and operation != "languages":
        return {
            "tool": "mm_tesseract",
            "status": "error",
            "error": "Tesseract is not installed. Install with: apt install tesseract-ocr",
        }
    
    try:
        if operation == "ocr":
            result = _ocr(
                image_path=args.get("image_path", ""),
                lang=args.get("lang", "eng"),
                config=args.get("config", ""),
                psm=args.get("psm"),
                oem=args.get("oem"),
            )
        
        elif operation == "ocr_data":
            result = _ocr_data(
                image_path=args.get("image_path", ""),
                lang=args.get("lang", "eng"),
            )
        
        elif operation == "detect_orientation":
            result = _detect_orientation(
                image_path=args.get("image_path", ""),
            )
        
        elif operation == "languages":
            result = _get_languages()
        
        elif operation == "pdf_to_text":
            result = _pdf_to_text(
                pdf_path=args.get("pdf_path", ""),
                lang=args.get("lang", "eng"),
                dpi=args.get("dpi", 300),
                first_page=args.get("first_page"),
                last_page=args.get("last_page"),
            )
        
        elif operation == "to_pdf":
            result = _ocr_to_searchable_pdf(
                image_path=args.get("image_path", ""),
                output_path=args.get("output_path", "output.pdf"),
                lang=args.get("lang", "eng"),
            )
        
        else:
            return {
                "tool": "mm_tesseract",
                "status": "error",
                "error": f"Unknown operation: {operation}",
            }
        
        return {"tool": "mm_tesseract", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "mm_tesseract", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "basic_ocr": {
            "operation": "ocr",
            "image_path": "document.png",
            "lang": "eng",
        },
        "ocr_with_config": {
            "operation": "ocr",
            "image_path": "receipt.jpg",
            "lang": "eng",
            "psm": 6,  # Assume uniform block of text
        },
        "get_bboxes": {
            "operation": "ocr_data",
            "image_path": "form.png",
            "lang": "eng",
        },
        "pdf_ocr": {
            "operation": "pdf_to_text",
            "pdf_path": "scanned_document.pdf",
            "lang": "eng",
            "dpi": 300,
        },
    }
