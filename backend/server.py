from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import pytesseract
import io
import traceback

app = FastAPI(title="Filevora lightweight OCR test", version="0.2.0")
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

def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image

def ocr_image(image: Image.Image) -> str:
    image = prepare_image(image)
    return pytesseract.image_to_string(image, lang="eng", config="--oem 1 --psm 3").strip()

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
        return JSONResponse({
            "ok": True,
            "requested_mode": "tesseract-lightweight",
            "engine": "Tesseract OCR (lightweight)",
            "ocr_model": "Tesseract English",
            "text": combined_text or "No readable text detected. Try a sharper, straight, well-lit notification image.",
            "artifacts": page_texts,
            "warning": "Experimental OCR only. Verify all file sizes, dimensions, and document rules against the original notification before using them in Form Ready."
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lightweight OCR failed: {str(exc)[:500]}")
