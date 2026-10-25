

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
    version="0.5.0"
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
MAX_IMAGE_SIDE = 2600

LABEL_PATTERNS = [
    ("photograph", re.compile(
        r"\b(?:photograph|passport\s*photo|photo\s*graph)\b", re.I)),
    ("signature", re.compile(r"\b(?:signature)\b", re.I)),
    ("marksheets", re.compile(
        r"\b(?:(?:10th|12th)\s*(?:&|and)?\s*)?marksheets?|"
        r"mark\s*sheets?|degrees?\s*(?:and|&)\s*mark\s*sheets?|"
        r"graduation|post\s*graduation\b", re.I)),
    ("identity_document", re.compile(
        r"\b(?:aadhaar|aadhar|photo\s*id|identity\s*card)\b", re.I)),
    ("application_form", re.compile(
        r"\b(?:signed\s*application\s*form|application\s*form)\b", re.I)),
    ("category_certificate", re.compile(
        r"\b(?:category\s*certificate|sc\s*/?\s*st|obc|bc-?\s*[ab]|"
        r"ews|non-?creamy(?:\s*layer)?|domicile|haryana\s*resident|"
        r"residence\s*certificate|dsc\s*/\s*osc)\b", re.I)),
    ("noc_experience", re.compile(
        r"\b(?:noc|experience\s*proof|experience\s*certificate)\b", re.I)),
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
    matches = []

    for kind, pattern in LABEL_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((
                kind, match.start(), match.end(), match.group(0)
            ))

    # Generic certificate matches are added only if not already covered
    # by a specific document label.
    for match in GENERIC_CERT_RE.finditer(text):
        covered = any(
            start <= match.start() < end
            for _, start, end, _ in matches
        )
        if not covered:
            matches.append((
                "certificate", match.start(),
                match.end(), match.group(0)
            ))

    matches.sort(key=lambda item: item[1])

    deduped = []
    for item in matches:
        kind, start, end, label = item

        if deduped and kind == deduped[-1][0]:
            if start - deduped[-1][2] < 35:
                continue

        deduped.append(item)

    return deduped


def extract_requirements(text: str):
    text = normalize_text(text)
    labels = find_labels(text)
    candidates = []
    seen = set()

    for i, (kind, start, label_end, label) in enumerate(labels):
        # Stop at the next document heading or after a short window.
        next_start = (
            labels[i + 1][1]
            if i + 1 < len(labels)
            else len(text)
        )
        end = min(next_start, start + 300)
        snippet = text[start:end].strip()

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

            kb = (
                int(round(value * 1024))
                if unit == "MB"
                else int(round(value))
            )

            sizes.append({
                "value": value,
                "unit": unit,
                "max_size_kb_approx": kb
            })

        formats = list(dict.fromkeys(
            m.group(1).upper()
            for m in FORMAT_RE.finditer(snippet)
        ))

        # Remove duplicate values while keeping their original order.
        unique_dimensions = []
        dimension_keys = set()
        for item in dimensions:
            key = (item["width_px"], item["height_px"])
            if key not in dimension_keys:
                dimension_keys.add(key)
                unique_dimensions.append(item)

        unique_sizes = []
        size_keys = set()
        for item in sizes:
            key = (item["value"], item["unit"])
            if key not in size_keys:
                size_keys.add(key)
                unique_sizes.append(item)

        candidate = {
            "document_type": kind,
            "label_found": label,
            "dimensions": unique_dimensions[:2],
            "allowed_formats_found": formats[:3],
            "file_size_limits_found": unique_sizes[:3],
            "source_snippet": snippet[:350],
            "status": "needs_review",
            "verified": False,
            "review_note": (
                "OCR-derived candidate. Verify against the original "
                "official notification before using."
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
                (s["value"], s["unit"])
                for s in candidate["file_size_limits_found"]
            )
        )

        if fingerprint not in seen:
            seen.add(fingerprint)
            candidates.append(candidate)

    browser_noise = bool(re.search(
        r"\b(?:google\.com/search|shopping|videos news forums|"
        r"AI overview|KB/S|sign in|settings)\b",
        text, re.I
    ))

    for item in candidates:
        if (
            not item["dimensions"]
            and item["document_type"] in ("photograph", "signature")
        ):
            item["review_note"] = (
                "No pixel dimensions found near this label; "
                "check the original table."
            )

        if len(item["file_size_limits_found"]) > 1:
            item["review_note"] += (
                " Multiple size values found nearby; verify which "
                "limit belongs to this document."
            )

    return {
        "schema_version": "0.3",
        "status": "candidates_only",
        "browser_or_page_noise_detected": browser_noise,
        "candidate_count": len(candidates),
        "note": (
            "OCR extraction only, not official verification. For best "
            "results upload a clear crop of the actual requirements table. "
            "Check every candidate against the official notification."
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
        "version": "0.5.0"
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
        raise HTTPException(
            status_code=400,
            detail="File is empty."
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Upload limit is 15 MB."
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
                    detail="Could not read this PDF."
                ) from exc

            for page_num in range(
                1,
                min(max(total_pages, 1), MAX_PDF_PAGES) + 1
            ):
                images = convert_from_bytes(
                    data,
                    dpi=130,
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

        extracted = extract_requirements(combined_text)

        return JSONResponse({
            "ok": True,
            "requested_mode": "tesseract-lightweight",
            "engine": "Tesseract OCR (lightweight)",
            "ocr_model": "Tesseract English",
            "text": combined_text or (
                "No readable text detected. Try a sharper, straight image."
            ),
            "artifacts": page_texts,
            "extracted_requirements": extracted,
            "warning": (
                "Experimental OCR extraction only. Verify every requirement "
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
