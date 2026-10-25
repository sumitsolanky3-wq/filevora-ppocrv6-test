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

app = FastAPI(title="Filevora lightweight OCR + requirements extraction", version="0.3.0")
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

SECTION_LABELS = [
    ("photograph", r"\b(?:photograph|photo\s*(?:graph)?|passport\s*photo|passport)\b"),
    ("signature", r"\b(?:signature|sign)\b"),
    ("marksheets", r"\b(?:degrees?\s*(?:and|&)?\s*mark\s*sheets?|marksheets?|mark\s*sheets?)\b"),
    ("identity_document", r"\b(?:aadhaar|aadhar|photo\s*id|identity\s*card)\b"),
    ("application_form", r"\b(?:signed\s*application\s*form|application\s*form)\b"),
    ("category_certificate", r"\b(?:category\s*certificate|sc\s*/?\s*st|obc|bc-?[ab]|ews|non-?creamy|domicile|residen(?:t|ce)\s*certificate)\b"),
    ("certificate", r"\b(?:certificate|certificates)\b"),
]

DIMENSION_RE = re.compile(
    r"(?<!\d)(\d{2,4})\s*(?:x|×|X)\s*(\d{2,4})\s*(?:px|pixels?)?",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"(?:up\s*to|max(?:imum)?\s*)?(\d+(?:\.\d+)?)\s*(KB|MB)\b",
    re.IGNORECASE,
)
FORMAT_RE = re.compile(r"\b(JPEG|JPG|PNG|PDF)\b", re.IGNORECASE)

def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image

def ocr_image(image: Image.Image) -> str:
    image = prepare_image(image)
    return pytesseract.image_to_string(
        image, lang="eng", config="--oem 1 --psm 3"
    ).strip()

def _find_section_spans(text: str):
    """Return likely document sections with nearby text; values remain unverified."""
    matches = []
    for kind, pattern in SECTION_LABELS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if kind == "certificate" and any(
                other_kind == "category_certificate" and other.start() <= match.start() < other.end()
                for other_kind, other in matches
            ):
                continue
            matches.append((kind, match))
    matches.sort(key=lambda item: item[1].start())

    deduped = []
    for kind, match in matches:
        if deduped and match.start() < deduped[-1][1].end() + 3:
            previous_kind, previous_match = deduped[-1]
            if len(kind) > len(previous_kind):
                deduped[-1] = (kind, match)
            continue
        deduped.append((kind, match))

    # OCR can split one heading such as "Aadhaar Card / Photo ID" into
    # multiple matches of the same category. Merge close duplicate labels.
    merged = []
    for kind, match in deduped:
        if merged and merged[-1][0] == kind:
            previous_match = merged[-1][1]
            gap = text[previous_match.end():match.start()]
            if len(gap) <= 35 and "\n" not in gap:
                continue
        merged.append((kind, match))

    spans = []
    for index, (kind, match) in enumerate(merged):
        start = match.start()
        end = merged[index + 1][1].start() if index + 1 < len(merged) else min(len(text), start + 900)
        spans.append((kind, text[start:end].strip()[:900]))
    return spans

def extract_requirements(text: str):
    """Extract candidates only. Every result requires human verification."""
    candidates = []
    seen = set()
    for kind, snippet in _find_section_spans(text):
        dimensions = [
            {"width_px": int(m.group(1)), "height_px": int(m.group(2))}
            for m in DIMENSION_RE.finditer(snippet)
        ]
        sizes = []
        for match in SIZE_RE.finditer(snippet):
            value = float(match.group(1))
            unit = match.group(2).upper()
            size_kb = int(round(value * 1024)) if unit == "MB" else int(round(value))
            sizes.append({"value": value, "unit": unit, "max_size_kb_approx": size_kb})
        formats = list(dict.fromkeys(m.group(1).upper() for m in FORMAT_RE.finditer(snippet)))
        candidate = {
            "document_type": kind,
            "dimensions": dimensions[:3],
            "allowed_formats_found": formats[:4],
            "file_size_limits_found": sizes[:4],
            "source_snippet": snippet[:500],
            "status": "needs_review",
            "verified": False,
        }
        fingerprint = (
            kind,
            tuple((d["width_px"], d["height_px"]) for d in dimensions[:3]),
            tuple(formats[:4]),
            tuple((s["value"], s["unit"]) for s in sizes[:4]),
        )
        if fingerprint not in seen:
            seen.add(fingerprint)
            candidates.append(candidate)

    return {
        "schema_version": "0.1",
        "status": "candidates_only",
        "note": "OCR-derived candidates may be wrong or incomplete. Verify each value against the original official notification before applying it to files.",
        "requirements": candidates,
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Filevora lightweight OCR test",
        "requested_mode": "tesseract-lightweight",
        "model_loaded": True,
        "load_error": None,
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.bin").name
    suffix = Path(name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Upload an image or PDF.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload limit is 15 MB for this lightweight test.")

    try:
        page_texts = []
        if suffix == ".pdf":
            try:
                info = pdfinfo_from_bytes(data)
                total_pages = int(info.get("Pages", 1))
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Could not read this PDF. Try a valid PDF or upload an image.") from exc

            for page_num in range(1, min(max(total_pages, 1), MAX_PDF_PAGES) + 1):
                images = convert_from_bytes(
                    data, dpi=110, first_page=page_num, last_page=page_num,
                    fmt="png", thread_count=1, grayscale=True
                )
                if images:
                    try:
                        page_texts.append({"page": page_num, "text": ocr_image(images[0])})
                    finally:
                        images[0].close()
        else:
            try:
                image = Image.open(io.BytesIO(data))
                try:
                    page_texts.append({"page": 1, "text": ocr_image(image)})
                finally:
                    image.close()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Could not read this image. Please upload a valid JPG or PNG.") from exc

        combined_text = "\n\n".join(
            f"--- Page {p['page']} ---\n{p['text']}" for p in page_texts
        ).strip()
        requirements = extract_requirements(combined_text)
        return JSONResponse({
            "ok": True,
            "requested_mode": "tesseract-lightweight",
            "engine": "Tesseract OCR (lightweight)",
            "ocr_model": "Tesseract English",
            "text": combined_text or "No readable text detected. Try a sharper, straight, well-lit notification image.",
            "artifacts": page_texts,
            "extracted_requirements": requirements,
            "warning": "Experimental OCR and candidate extraction only. All dimensions, formats, and file-size limits need verification against the original official notification before use."
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lightweight OCR failed: {str(exc)[:500]}")
