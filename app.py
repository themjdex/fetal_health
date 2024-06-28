import sys
import logging
import os
import pandas as pd
import uvicorn
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from src.models.model_fit_predict import predict_model
from main_pipeline import data_preprocessing

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# class AddHealthMetrics(BaseModel):
#     data: list
    # features: list

app = FastAPI()


@app.get('/')
def main() -> str:
    """
    GET-запрос основной точки входа 

    :return: HTML-заголовок
    """
    return '<h1>Предсказание патологий плода</h1>'


def load_model(training_pipeline_params: TrainingPipelineParams) -> RandomForestClassifier:
    """
    Функция загружает сохраненную модель

    :param training_pipeline_params: параметры с путем хранения модели
    :return: объект модели
    """
    return joblib.load(training_pipeline_params.output_model_path)



def check_model(training_pipeline_params: TrainingPipelineParams) -> None:
    """
    Функция проверяет наличие модели

    :param training_pipeline_params: параметры с путем хранения модели
    :return: None
    """
    model = load_model(training_pipeline_params)
    if model is None:
        logger.error('app/check_model models are None')
        raise HTTPException(status_code=500, detail='Models are unavailable')


def check_schema(features: list, training_pipeline_params: TrainingPipelineParams) -> None:
    """
    Функция проверяет, что полученные признаки совпадают с признаками модели

    :param features: список признаков
    :param training_pipeline_params: список признаков из конфига
    :return: None
    """
    if not set(training_pipeline_params.feature_params.features).issubset(
        set(features)
    ):
        logger.error('app/check_schema missing columns')
        raise HTTPException(
            status_code=400, detail=f'Missing features in schema {features}'
        )


def make_predict(
    model: RandomForestClassifier,
    training_pipeline_params: TrainingPipelineParams,
) -> list:
    """
    Функция загружает полученный датасет и делает предикты

    :param data: CSV-датасет, для которого нужны предсказания
    :param features: список признаков
    :param model: обученная модель
    :param training_pipeline_params: конфиг обучения
    :return: список предсказаний для каждого объекта
    """

    # загрузка датасета
    df = pd.read_csv(training_pipeline_params.input_test_data_path)

    # проверка фичей по схеме
    check_schema(df.columns, training_pipeline_params)

    # удаляем таргет
    df = df.drop('fetal_health', axis=1)

    # минимальная предобработка данных
    df = data_preprocessing(df, training_pipeline_params)
    # создание предиктов
    predictions = predict_model(model, df)

    logger.info(f'predictions: {predictions}')

    return predictions.tolist()


@app.post('/predict')
def predict():
    """
    Функция обработки POST-запроса на получение предикта
    """
    logger.info('app/predict run')

    config_path = 'config/config.yaml'
    training_pipeline_params = read_training_pipeline_params(config_path)

    logger.info(f'app/predict training_pipeline_params: {training_pipeline_params}')

    check_model(training_pipeline_params)
    logger.info('app/predict check_models passed')

    model = load_model(training_pipeline_params)
    # return 'a'
    return make_predict(
        model, training_pipeline_params
    )


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True, port=os.getenv("PORT", 8000))
