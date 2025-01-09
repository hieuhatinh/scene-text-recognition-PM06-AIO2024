from fastapi import FastAPI
from controller import ocr

app = FastAPI()

app.include_router(ocr.ocr_router)
