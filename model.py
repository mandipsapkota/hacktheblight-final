from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from model import predict

app = FastAPI()

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    try:
        predictions = predict(image)
        return {"predictions": predictions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
