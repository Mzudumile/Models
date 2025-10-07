from fastapi import FastAPI, UploadFile, File, Form
from ofatest import *  # import your wrapper

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    task: str = Form(...)
):
    # Read file into memory
    file_bytes = await file.read()
    image_buffer = BytesIO(file_bytes)

    # Call OFA model directly with in-memory image
    description = ofa_model(image_buffer, prompt=task)

    return {"description": description}

@app.get('/')
def root():
    return {'Hello':'world'}