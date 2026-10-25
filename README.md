# Filevora — PP-OCRv6 + PP-StructureV3 isolated test kit

This is a **separate test project**. It does not change the existing Filevora website or OCR package.

## What it tests
- PP-StructureV3 document/table layout parsing
- Explicit PP-OCRv6 small text detection + recognition model names
- Raw OCR/table output for manual comparison against the source notification

It does **not** claim accuracy until you run it on the original notification and inspect the output.

## Important
The backend downloads model files on first use and needs a compatible Python/PaddlePaddle environment and enough RAM/CPU. A free hosting tier may restart, run slowly, or run out of memory. This is not a Cloudflare Pages-only feature; the Python backend must be hosted separately.

## Run locally
Python 3.10/3.11 recommended.

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `test.html` in a browser and set backend URL to `http://127.0.0.1:8000`.
If opening the file directly is blocked by browser CORS/file-origin behavior, serve the folder with:
```bash
python -m http.server 8080
```
Then open `http://127.0.0.1:8080/test.html`.

## Deploy as a separate test service
The `Dockerfile` and `render.yaml` are included for a separate backend service. Deploy this folder as its own project/service; do not replace your live Filevora Pages deployment.

## Test procedure
1. Upload the exact notification screenshot or PDF.
2. Press **Run layout + OCR test**.
3. Compare the raw result against the source. For this screenshot, expected visible values include:
   - Passport Photograph: 50–500 KB; 132 × 170 to 160 × 204 px; 3.5 × 4.5 cm.
   - Signature: 50–500 KB; 132 × 57 to 160 × 68 px; 4.0 × 2.0 cm.
   - Certificates: usually 10–200 KB in this screenshot; some specific posts may allow up to 2 MB.
4. Do not connect this to Form Ready until the output is verified on real notifications.

## Model fallback
Set environment variable `OCR_MODEL_MODE=default` to use the PP-StructureV3 default OCR models if explicit PP-OCRv6 model names are not supported by the installed PaddleOCR version. Default mode is `v6`.
