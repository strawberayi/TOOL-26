from fastapi import FastAPI, UploadFile, File
from masking_ita import MaskingITAProcessor

app = FastAPI()

processor = MaskingITAProcessor()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image_rgb = processor.load_image_bytes(contents)
        result = processor.process_image(image_rgb, filename=file.filename or "upload")
    except Exception as exc:
        from masking_ita import MaskingITAResult, ProcessingStatus
        result = MaskingITAResult(
            filename=file.filename or "upload",
            status=ProcessingStatus.IMAGE_FAILED.value,
            failure_reason=str(exc),
            user_message="Invalid image; please retake image.",
        )
    return result.to_dict()
