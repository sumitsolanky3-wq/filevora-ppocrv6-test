from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, json, os, traceback

app = FastAPI(title="Filevora isolated PP-OCRv6 layout test", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)
_pipeline = None
_pipeline_error = None
MODEL_MODE = os.getenv("OCR_MODEL_MODE", "v6").strip().lower()

def get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    if _pipeline_error:
        raise RuntimeError(_pipeline_error)
    from paddleocr import PPStructureV3
    common = dict(use_doc_orientation_classify=False, use_doc_unwarping=False)
    if MODEL_MODE == "default":
        _pipeline = PPStructureV3(**common)
    else:
        # Explicit PP-OCRv6 small detector + recognizer in the layout pipeline.
        # If this PaddleOCR release does not accept these model names, set OCR_MODEL_MODE=default.
        _pipeline = PPStructureV3(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            **common
        )
    return _pipeline

def read_json_files(folder):
    results = []
    for p in sorted(Path(folder).rglob("*.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results

def read_markdown_files(folder):
    texts = []
    for p in sorted(Path(folder).rglob("*.md")):
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return "\n\n".join(texts)

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Filevora isolated OCR test",
        "requested_mode": MODEL_MODE,
        "model_loaded": _pipeline is not None,
        "load_error": _pipeline_error
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.bin").name
    suffix = Path(name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}:
        raise HTTPException(400, "Upload an image or PDF.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty.")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(400, "Prototype limit is 25 MB.")
    try:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / ("upload" + suffix)
            inp.write_bytes(data)
            out = Path(td) / "out"
            out.mkdir()
            pipe = get_pipeline()
            results = pipe.predict(input=str(inp))
            for i, res in enumerate(results):
                try:
                    res.save_to_json(save_path=str(out / f"result_{i}.json"))
                except Exception:
                    pass
                try:
                    res.save_to_markdown(save_path=str(out / f"result_{i}.md"))
                except Exception:
                    pass
            md = read_markdown_files(out)
            artifacts = read_json_files(out)
            if not md:
                md = json.dumps(artifacts, ensure_ascii=False, indent=2) if artifacts else "\n".join(str(r) for r in results)
            return JSONResponse({
                "ok": True,
                "requested_mode": MODEL_MODE,
                "engine": "PaddleOCR PP-StructureV3",
                "ocr_model": "PP-OCRv6 small det+rec (requested)" if MODEL_MODE != "default" else "PP-StructureV3 default OCR models",
                "text": md,
                "artifacts": artifacts,
                "warning": "Experimental test only. Verify every value against the original notification before using it in Form Ready."
            })
    except Exception as e:
        traceback.print_exc()
        message = str(e)[:900]
        if MODEL_MODE != "default":
            message += " | If the installed PaddleOCR build rejects PP-OCRv6 model names, redeploy with OCR_MODEL_MODE=default."
        raise HTTPException(500, message)
