from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import pytesseract
import io
import re
import traceback

app = FastAPI(title="Filevora OCR Requirement Test", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 3
MAX_IMAGE_SIDE = 2600

LABEL_PATTERNS = [
    ("photograph", re.compile(
        r"\b(?:photograph|passport\s*photo|photo\s*graph)\b", re.I)),
    ("signature", re.compile(r"\bsignature\b", re.I)),
    ("marksheets", re.compile(
        r"\b(?:10th|12th|matriculation|marksheets?|mark\s*sheets?|"
        r"degree\s*(?:and|&)\s*marksheets?|graduation|post\s*graduation)\b",
        re.I)),
    ("identity_document", re.compile(
        r"\b(?:aadhaar|aadhar|photo\s*id|identity\s*card)\b", re.I)),
    ("application_form", re.compile(
        r"\b(?:signed\s*application\s*form|application\s*form)\b", re.I)),
    ("category_certificate", re.compile(
        r"\b(?:category\s*certificate|sc\s*/?\s*st|obc|bc-?\s*[ab]|"
        r"ews|non-?creamy(?:\s*layer)?|domicile|haryana\s*resident|"
        r"residence\s*certificate|dsc\s*/\s*osc)\b", re.I)),
    ("noc_experience", re.compile(
        r"\b(?:noc|experience\s*(?:proof|certificate))\b", re.I)),
]

GENERIC_CERT_RE = re.compile(r"\bcertificates?\b", re.I)
DIMENSION_RE = re.compile(
    r"(?<!\d)(\d{2,4})\s*(?:x|×)\s*(\d{2,4})\s*(?:px|pixels?)\b", re.I)
SIZE_RE = re.compile(
    r"\b(?:up\s*to|max(?:imum)?\s*)?(\d+(?:\.\d+)?)\s*(KB|MB)\b", re.I)
FORMAT_RE = re.compile(r"\b(JPEG|JPG|PNG|PDF)\b", re.I)


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_labels(text: str):
    """Find specific document labels and suppress ambiguous generic certificate labels."""
    found = []

    for kind, pattern in LABEL_PATTERNS:
        for match in pattern.finditer(text):
            found.append({
                "kind": kind,
                "start": match.start(),
                "end": match.end(),
                "label": match.group(0)
            })

    # Do not create generic "certificate" cards from every occurrence of
    # the word certificate. They are ambiguous and often duplicate a
    # specific document type such as category_certificate or noc_experience.
    found.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))

    result = []
    for item in found:
        if result and item["kind"] == result[-1]["kind"]:
            between = text[result[-1]["end"]:item["start"]]
            # OCR often repeats the same heading nearby. Ignore close repeats.
            if len(between) < 90:
                continue
        result.append(item)

    return result


def unique_keep_order(items, key):
    seen = set()
    result = []

    for item in items:
        value = key(item)
        if value not in seen:
            seen.add(value)
            result.append(item)

    return result


def extract_requirements(text: str):
    text = normalize_text(text)
    labels = find_labels(text)
    candidates = []

    for index, item in enumerate(labels):
        start = item["start"]
        next_start = (
            labels[index + 1]["start"]
            if index + 1 < len(labels)
            else len(text)
        )

        snippet = text[start:min(next_start, start + 260)].strip()

        dimensions = [
            {
                "width_px": int(match.group(1)),
                "height_px": int(match.group(2))
            }
            for match in DIMENSION_RE.finditer(snippet)
        ]

        dimensions = unique_keep_order(
            dimensions,
            lambda value: (value["width_px"], value["height_px"])
        )

        sizes = []
        for match in SIZE_RE.finditer(snippet):
            value = float(match.group(1))
            unit = match.group(2).upper()
            approx_kb = (
                int(round(value * 1024))
                if unit == "MB"
                else int(round(value))
            )

            sizes.append({
                "value": value,
                "unit": unit,
                "max_size_kb_approx": approx_kb
            })

        sizes = unique_keep_order(
            sizes,
            lambda value: (value["value"], value["unit"])
        )

        formats = unique_keep_order(
            [
                match.group(1).upper()
                for match in FORMAT_RE.finditer(snippet)
            ],
            lambda value: value
        )

        candidate = {
            "id": f"{item['kind']}-{index + 1}",
            "document_type": item["kind"],
            "label_found": item["label"],
            "dimensions": dimensions[:2],
            "allowed_formats_found": formats[:3],
            "file_size_limits_found": sizes[:3],
            "source_snippet": snippet[:320],
            "status": "needs_review",
            "verified": False,
            "review_note": (
                "Check this candidate against the official "
                "notification before using."
            )
        }

        if item["kind"] in ("photograph", "signature") and not dimensions:
            candidate["review_note"] = (
                "No pixel dimensions found near this heading. "
                "Check the original table."
            )

        if len(sizes) > 1:
            candidate["review_note"] += (
                " Multiple size limits nearby; verify which applies."
            )

        candidates.append(candidate)

    # Merge exact duplicate OCR candidates.
    merged = []
    by_key = {}

    for candidate in candidates:
        key = (
            candidate["document_type"],
            tuple(
                (dimension["width_px"], dimension["height_px"])
                for dimension in candidate["dimensions"]
            ),
            tuple(candidate["allowed_formats_found"]),
            tuple(
                (size["value"], size["unit"])
                for size in candidate["file_size_limits_found"]
            )
        )

        if key not in by_key:
            by_key[key] = len(merged)
            merged.append(candidate)
        else:
            old_index = by_key[key]
            old = merged[old_index]

            old_score = (
                len(old["dimensions"])
                + len(old["allowed_formats_found"])
                + len(old["file_size_limits_found"])
            )

            new_score = (
                len(candidate["dimensions"])
                + len(candidate["allowed_formats_found"])
                + len(candidate["file_size_limits_found"])
            )

            if new_score > old_score:
                merged[old_index] = candidate

    # Remove empty generic certificate cards caused by table headings.
    merged = [
        candidate for candidate in merged
        if not (
            candidate["document_type"] == "certificate"
            and not candidate["dimensions"]
            and not candidate["allowed_formats_found"]
            and not candidate["file_size_limits_found"]
        )
    ]

    noise = bool(re.search(
        r"\b(?:google\.com/search|shopping|videos news forums|"
        r"AI overview|KB/S|sign in|settings)\b",
        text,
        re.I
    ))

    return {
        "schema_version": "0.5",
        "status": "candidates_only",
        "browser_or_page_noise_detected": noise,
        "candidate_count": len(merged),
        "note": (
            "OCR-derived candidates only. Verify all values "
            "against the official notification."
        ),
        "requirements": merged
    }


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS
    )
    return image


def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(
        prepare_image(image),
        lang="eng",
        config="--oem 1 --psm 6"
    ).strip()


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Filevora lightweight OCR test",
        "requested_mode": "tesseract-lightweight",
        "model_loaded": True,
        "load_error": None,
        "version": "0.7.0"
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()

    allowed = {
        ".png", ".jpg", ".jpeg", ".webp",
        ".bmp", ".tif", ".tiff", ".pdf"
    }

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Upload an image or PDF."
        )

    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Upload limit is 15 MB."
        )

    try:
        pages = []

        if suffix == ".pdf":
            try:
                info = pdfinfo_from_bytes(data)
                page_count = int(info.get("Pages", 1))
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read this PDF."
                ) from exc

            for page_number in range(
                1, min(max(page_count, 1), MAX_PDF_PAGES) + 1
            ):
                images = convert_from_bytes(
                    data,
                    dpi=130,
                    first_page=page_number,
                    last_page=page_number,
                    fmt="png",
                    thread_count=1,
                    grayscale=True
                )

                if images:
                    try:
                        pages.append({
                            "page": page_number,
                            "text": ocr_image(images[0])
                        })
                    finally:
                        images[0].close()

        else:
            try:
                image = Image.open(io.BytesIO(data))
                try:
                    pages.append({
                        "page": 1,
                        "text": ocr_image(image)
                    })
                finally:
                    image.close()
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read this image."
                ) from exc

        full_text = "\n\n".join(
            f"--- Page {page['page']} ---\n{page['text']}"
            for page in pages
        ).strip()

        return JSONResponse({
            "ok": True,
            "requested_mode": "tesseract-lightweight",
            "engine": "Tesseract OCR (lightweight)",
            "ocr_model": "Tesseract English",
            "text": (
                full_text
                or "No readable text detected. Try a clearer, straight image."
            ),
            "artifacts": pages,
            "extracted_requirements": extract_requirements(full_text),
            "warning": (
                "Experimental OCR extraction. Verify all values "
                "against the official notification."
            )
        })

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lightweight OCR failed: {str(exc)[:500]}"
        )

# Filevora Form Ready - first image processing endpoint
from fastapi.responses import Response as BinaryResponse


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    width: int = 100,
    height: int = 120,
    max_kb: int = 200
):
    filename = Path(file.filename or "upload.jpg").name
    suffix = Path(filename).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, WEBP or BMP image."
        )

    if not (1 <= width <= 3000 and 1 <= height <= 3000):
        raise HTTPException(
            status_code=400,
            detail="Width and height must be between 1 and 3000 pixels."
        )

    if not (1 <= max_kb <= 10240):
        raise HTTPException(
            status_code=400,
            detail="Maximum file size must be between 1 and 10240 KB."
        )

    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload limit is 15 MB.")

    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize(
                (width, height),
                Image.Resampling.LANCZOS
            )

            output = None
            limit_bytes = max_kb * 1024

            for quality in range(90, 9, -5):
                buffer = io.BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=True
                )
                candidate = buffer.getvalue()
                buffer.close()

                if len(candidate) <= limit_bytes:
                    output = candidate
                    break

            image.close()

            if output is None:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "This image cannot meet the requested size "
                        "at these exact dimensions. Use a higher KB limit."
                    )
                )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not process this image."
        ) from exc

    ready_name = f"{Path(filename).stem}_ready.jpg"

    return BinaryResponse(
        content=output,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{ready_name}"',
            "X-Image-Width": str(width),
            "X-Image-Height": str(height),
            "X-File-Size-Bytes": str(len(output))
        }
    )
