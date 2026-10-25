
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

app = FastAPI(
    title="Filevora lightweight OCR + requirements extraction",
    version="0.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 3
MAX_IMAGE_SIDE = 2200

SECTION_PATTERNS = [
    ("photograph", re.compile(
        r"\b(?:photograph|photo\s*graph|passport\s*photo)\b", re.I)),
    ("signature", re.compile(r"\b(?:signature|sign)\b", re.I)),
    ("marksheets", re.compile(
        r"\b(?:10th|12th|marksheets?|mark\s*sheets?|degree\s*certificates?)\b", re.I)),
    ("identity_document", re.compile(
        r"\b(?:aadhaar|aadhar|photo\s*id|identity\s*card)\b", re.I)),
    ("application_form", re.compile(
        r"\b(?:signed\s*application\s*form|application\s*form)\b", re.I)),
    ("category_certificate", re.compile(
        r"\b(?:category\s*certificate|sc\s*/?\s*st|obc|bc-?[ab]|ews|non-?creamy|domicile|residen(?:t|ce)\s*certificate)\b", re.I)),
    ("certificate", re.compile(r"\b(?:certificate|certificates)\b", re.I)),
]

DIMENSION_RE = re.compile(
    r"(?<!\d)(\d{2,4})\s*(?:x|×)\s*(\d{2,4})\s*(?:px|pixels?)\b",
    re.I
)
SIZE_RE = re.compile(
    r"\b(?:up\s*to|max(?:imum)?\s*)?(\d+(?:\.\d+)?)\s*(KB|MB)\b",
    re.I
)
FORMAT_RE = re.compile(r"\b(JPEG|JPG|PNG|PDF)\b", re.I)


def _clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _label_occurrences(text: str):
    found = []

    for kind, pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            if kind == "certificate" and any(
                other_kind == "category_certificate"
                and other_match.start() <= match.start() < other_match.end()
                for other_kind, other_match in found
            ):
                continue

            found.append((kind, match))

    found.sort(key=lambda item: item[1].start())

    deduped = []
    for kind, match in found:
        if deduped and kind == deduped[-1][0]:
            gap = text[deduped[-1][1].end():match.start()]
            if len(gap) < 45:
                continue

        deduped.append((kind, match))

    return deduped


def extract_requirements(text: str):
    """Create OCR-derived candidates, not verified official requirements."""
    text = _clean_ocr_text(text)
    labels = _label_occurrences(text)
    candidates = []
    seen = set()

    for index, (kind, match) in enumerate(labels):
        next_start = (
            labels[index + 1][1].start()
            if index + 1 < len(labels)
            else len(text)
        )

        # Limit each candidate to its local section.
        end = min(next_start, match.end() + 240)
        snippet = text[match.start():end].strip()

        dimensions = [
            {
                "width_px": int(m.group(1)),
                "height_px": int(m.group(2))
            }
            for m in DIMENSION_RE.finditer(snippet)
        ]

        sizes = []
        for m in SIZE_RE.finditer(snippet):
            value = float(m.group(1))
            unit = m.group(2).upper()

            size_kb = (
                int(round(value * 1024))
                if unit == "MB"
                else int(round(value))
            )

            sizes.append({
                "value": value,
                "unit": unit,
                "max_size_kb_approx": size_kb
            })

        formats = list(dict.fromkeys(
            m.group(1).upper()
            for m in FORMAT_RE.finditer(snippet)
        ))

        candidate = {
            "document_type": kind,
            "dimensions": dimensions[:1],
            "allowed_formats_found": formats[:3],
            "file_size_limits_found": sizes[:2],
            "source_snippet": snippet[:300],
            "status": "needs_review",
            "verified": False,
            "review_note": (
                "Verify this candidate against the original official "
                "notification. OCR may miss or misread values."
            )
        }

        fingerprint = (
            kind,
            tuple(
                (d["width_px"], d["height_px"])
                for d in candidate["dimensions"]
            ),
            tuple(candidate["allowed_formats_found"]),
            tuple(
                (z["value"], z["unit"])
                for z in candidate["file_size_limits_found"]
            ),
            re.sub(r"\W+", "", snippet.lower())[:80]
        )

        if fingerprint not in seen:
            seen.add(fingerprint)
            candidates.append(candidate)

    browser_noise = bool(re.search(
        r"\b(?:google\.com/search|shopping|videos news forums|AI overview|KB/S)\b",
        text,
        re.I
    ))

    return {
        "schema_version": "0.2",
        "status": "candidates_only",
        "browser_or_page_noise_detected": browser_noise,
        "note": (
            "Conservative OCR-assisted extraction, not AI verification. "
            "If browser/page noise is detected, upload a close crop of "
            "the official requirements table. Verify every value before use."
        ),
        "requirements": candidates
    }


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS
    )
    return image


def ocr_image(image: Image.Image) -> str:
    image = prepare_image(image)
    return pytesseract.image_to_string(
        image,
        lang="eng",
        config="--oem 1 --psm 3"
    ).strip()


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Filevora lightweight OCR test",
        "requested_mode": "tesseract-lightweight",
        "model_loaded": True,
        "load_error": None
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.bin").name
    suffix = Path(name).suffix.lower()

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
        raise HTTPException(
            status_code=400,
            detail="File is empty."
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Upload limit is 15 MB for this lightweight test."
        )

    try:
        page_texts = []

        if suffix == ".pdf":
            try:
                info = pdfinfo_from_bytes(data)
                total_pages = int(info.get("Pages", 1))
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read this PDF. Try a valid PDF or upload an image."
                ) from exc

            for page_num in range(
                1,
                min(max(total_pages, 1), MAX_PDF_PAGES) + 1
            ):
                images = convert_from_bytes(
                    data,
                    dpi=110,
                    first_page=page_num,
                    last_page=page_num,
                    fmt="png",
                    thread_count=1,
                    grayscale=True
                )

                if images:
                    try:
                        page_texts.append({
                            "page": page_num,
                            "text": ocr_image(images[0])
                        })
                    finally:
                        images[0].close()

        else:
            try:
                image = Image.open(io.BytesIO(data))
                try:
                    page_texts.append({
                        "page": 1,
                        "text": ocr_image(image)
                    })
                finally:
                    image.close()

            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read this image. Upload a valid image."
                ) from exc

        combined_text = "\n\n".join(
            f"--- Page {p['page']} ---\n{p['text']}"
            for p in page_texts
        ).strip()

        requirements = extract_requirements(combined_text)

        return JSONResponse({
            "ok": True,
            "requested_mode": "tesseract-lightweight",
            "engine": "Tesseract OCR (lightweight)",
            "ocr_model": "Tesseract English",
            "text": combined_text or (
                "No readable text detected. Try a sharper, straight image."
            ),
            "artifacts": page_texts,
            "extracted_requirements": requirements,
            "warning": (
                "Experimental OCR and candidate extraction only. "
                "Verify all dimensions, formats, and size limits against "
                "the original official notification before use."
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
