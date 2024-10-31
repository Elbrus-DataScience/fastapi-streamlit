import logging
import random
from contextlib import asynccontextmanager

import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.model_func import (
    class_id_to_label, load_pt_model,
    load_sklearn_model, transform_image
)

logger = logging.getLogger('uvicorn.info')


# Определение класса ответа для классификации изображений
class ImageResponse(BaseModel):
    class_name: str  # Название класса, например, dog, cat и т.д.
    class_index: int # Индекс класса из файла с индексами ImageNet

# Определение класса запроса для классификации текста
class TextInput(BaseModel):
    text: str  # Текст, введенный пользователем для классификации

# Определение класса ответа для классификации текста
class TextResponse(BaseModel):
    label: str  # Метка класса, например, positive или negative
    prob: float # Вероятность, связанная с меткой

# Определение класса запроса для табличных данных
class TableInput(BaseModel):
    feature1: float # Первая числовая характеристика
    feature2: float # Вторая числовая характеристика

# Определение класса ответа для табличных данных
class TableOutput(BaseModel):
    prediction: float # Предсказанное значение (например, 1 или 0)


pt_model = None  # Глобальная переменная для PyTorch модели
sk_model = None  # Глобальная переменная для Sklearn модели

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global pt_model
    global sk_model
    # Загрузка PyTorch модели
    pt_model = load_pt_model()
    logger.info('Torch model loaded')
    # Загрузка Sklearn модели
    sk_model = load_sklearn_model()
    logger.info('Sklearn model loaded')
    yield
    # Удаление моделей и освобождение ресурсов
    del pt_model, sk_model

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return 'Hello FastAPI!'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    """
    Эндпоинт для классификации изображений.
    Принимает файл изображения, обрабатывает его, делает предсказание и возвращает название и индекс класса.
    """
    # Открытие изображения
    image = PIL.Image.open(file.file)
    # Предобработка изображения
    adapted_image = transform_image(image)
    # Логирование формы обработанного изображения
    logger.info(f'{adapted_image.shape}')
    # Предсказание класса изображения
    with torch.inference_mode():
        pred_index = pt_model(adapted_image).numpy().argmax()
    # Преобразование индекса в название класса
    imagenet_class = class_id_to_label(pred_index)
    # Формирование ответа
    response = ImageResponse(
        class_name=imagenet_class,
        class_index=pred_index
    )
    return response

@app.post('/clf_table')
def predict(x: TableInput):
    """
    Эндпоинт для классификации табличных данных.
    Принимает значения признаков и возвращает предсказание модели.
    """
    # Преобразование признаков в массив и предсказание
    prediction = sk_model.predict(np.array([x.feature1, x.feature2]).reshape(1, 2))
    # Формирование ответа
    result = TableOutput(prediction=prediction[0])
    return result

@app.post('/clf_text')
def clf_text(data: TextInput):
    """
    Эндпоинт для классификации текста.
    Случайно генерирует метку класса и вероятность для демонстрационных целей.
    """
    # Генерация случайного класса и вероятности
    pred_class = random.choice(['positive', 'negative'])
    probability = random.random()
    # Формирование ответа
    response = TextResponse(
        label=pred_class,
        prob=probability
    )
    return response

if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)

##### 
# Проверка с помощью утилиты cURL:
# curl -X POST "http://127.0.0.1:8000/classify_image/" -L -H "Content-Type: multipart/form-data" -F "file=@dog.jpeg;type=image/jpeg"
#####
