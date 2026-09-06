from fastapi import FastAPI, UploadFile, File
from masking_ita import MaskingITAProcessor
import numpy as np
import cv2

app = FastAPI()

processor = MaskingITAProcessor()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = processor.process_image(
        image_rgb,
        filename=file.filename
    )

    return result.to_dict()