import logging
import random
from contextlib import asynccontextmanager

import PIL

logger = logging.getLogger('uvicorn.info')

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.model_func import (
    class_id_to_label, load_pt_model,
    load_sklearn_model, transform_image
    )


# Create class of answer: class name and class index
class ImageResponse(BaseModel):
    class_name: str # dog, cat, etc
    class_index: int # class index from imagenet json file

class TextInput(BaseModel):
    text: str   # some user text to classify

class TextResponse(BaseModel):
    label: str  # positive/negative
    prob: float # some probability

class TableInput(BaseModel):
    feature1: float # float features
    feature2: float # float features

class TableOutput(BaseModel):
    prediction: float # class 1 or 0


pt_model = None 
sk_model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global pt_model
    global sk_model
    pt_model = load_pt_model()
    logger.info('Torch model loaded')
    sk_model = load_sklearn_model()
    logger.info('Sklearn model loaded')
    yield
    # Clean up the ML models and release the resources
    del pt_model, sk_model


app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    return 'Hello!'


@app.post('/clf_image')
def classify_image(file: UploadFile):
    # open image
    image = PIL.Image.open(file.file) 
    # preprocess image
    adapted_image = transform_image(image) 
    # log 
    logger.info(f'{adapted_image.shape}')
    # predict 
    with torch.inference_mode():
        pred_index = pt_model(adapted_image).numpy().argmax()
    # convert index to class
    imagenet_class = class_id_to_label(pred_index)
    # make correct response
    response = ImageResponse(
        class_name=imagenet_class,
        class_index=pred_index
    )
    return response

@app.post('/clf_table')
def predict(x: TableInput):
    logger.warning(np.array([x.feature1, x.feature2]).reshape(1, 2).shape)
    prediction = sk_model.predict(np.array([x.feature1, x.feature2]).reshape(1, 2))
    result = TableOutput(prediction=prediction[0])
    return result

@app.post('/clf_text')
def clf_text(data: TextInput):
    # generate fake class and probability
    pred_class = random.choice(['positive', 'negative'])
    probability = random.random()
    response = TextResponse(
        label=pred_class,
        prob=probability
    )
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)



##### 
# check via cURL util:
# curl -X POST "http://127.0.0.1:8000/classify_image/" -L -H  "Content-Type: multipart/form-data" -F "file=@dog.jpeg;type=image/jpeg"
####