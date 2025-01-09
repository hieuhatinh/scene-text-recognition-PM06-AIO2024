from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Response
from PIL import Image
import io
import os
import tempfile
from models.ocr import ocr_service
import requests

app = FastAPI()

ocr_router = APIRouter(
    prefix="/ocr",
    tags=['ocr']
)
# class APIIngress:
#     def __init__(self) -> None:
#         pass


async def process(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(file)
            temp_file_path = temp_file.name

        prediction_result = await ocr_service.process_image(file)

        image = Image.open(temp_file_path)
        draw_predictions = await ocr_service.draw_prediction(image, prediction_result)

        # Convert annotated image to bytes
        file_stream = io.BytesIO()
        draw_predictions.save(file_stream, format="PNG")
        file_stream.seek(0)

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return prediction_result, file_stream
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {e}")


@ocr_router.post('/upload/')
async def ocr_upload(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, detail='File phải là 1 ảnh')
        file = await file.read()
        predictions, file_stream = await process(file)
        return Response(
            content=file_stream.getvalue(),
            media_type='image/png',
            headers={'X-predictions': str(predictions)}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=e
        )


@ocr_router.get('/url/')
async def ocr_img_url(image_url: str):
    try:
        response = requests.get(image_url)
        predictions, file_stream = await process(response.content)
        return Response(
            content=file_stream.getvalue(),
            media_type='image/png',
            headers={'X-predictions': str(predictions)}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=e
        )
