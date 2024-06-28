import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score
)
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Union


from src.entities.train_params import TrainingParams

Classifier = Union[RandomForestClassifier]


def train_model(
    features: pd.DataFrame, 
    target: pd.Series, 
    train_params: TrainingParams
) -> Classifier:
    """
    Обучение случайного леса и сохранение модели

    :param features: признаки датасета
    :param target: целевой признак
    :param train_params: датакласс с параметрами обучения
    :return Classifier: обученная модель классификатора
    """

    model = RandomForestClassifier(
        class_weight=train_params.class_weight,
        random_state=train_params.random_state,
        n_estimators=train_params.n_estimators,
        max_depth=train_params.max_depth,
        min_samples_split=train_params.min_samples_split,
        min_samples_leaf=train_params.min_samples_leaf,
        n_jobs=train_params.n_jobs
    )
    model.fit(features, target)
    return model


def predict_model(
    model: Classifier, 
    features: pd.DataFrame
) -> np.ndarray:
    """
    Получение предсказаний на тестовой выборке

    :params model: обученная модель случайного леса
    :params features: датафрейм с тестовыми признаками
    :return: массив Numpy с предсказанными ответами 
    """
    preds = model.predict(features)
    return preds

# продолжить завтра отсюда!

def evaluate_model(
    predicts: np.ndarray, 
    target: pd.Series
) -> Dict[str, float]:
    """Получение взвешенных оценок предсказаний
    (F1, recall и precision)

    :param predicts: полученные предсказания
    :param target: истинные метки
    :return: словарь с метриками
    """
    return {
        'f1_score': f1_score(target, predicts, average='weighted'),
        'precision': precision_score(target, predicts, average='weighted'),
        'recall': recall_score(target, predicts, average='weighted')
    }


def serialize_model(model, output: str) -> str:
    """
    Сохранение дампа обученной модели по указанному пути

    :param model: объект обученной модели
    :param output: путь для сохранения
    :return: путь, где была сохранена модель
    """
    with open(output, "wb") as file:
        joblib.dump(model, file)
    return output