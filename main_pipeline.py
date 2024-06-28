import json
import logging
import sys
import argparse
import pandas as pd

from src.data.make_dataset import (read_data, 
                                   drop_duplicates, 
                                   fill_na, 
                                   drop_useless_features, 
                                   split_features_target, 
                                   split_train_test_data)

from src.entities.train_pipeline_params import (
    TrainingPipelineParams, 
    read_training_pipeline_params
    )


from src.models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def create_schema_params(config_path: str) -> TrainingPipelineParams:
    """
    Функция создает схему параметров обучения

    :param config_path: путь к конфигу обучения
    :return: схема TrainingPipelineParams
    """
    return read_training_pipeline_params(config_path)


def data_preprocessing(data: pd.DataFrame, training_pipeline_params: TrainingPipelineParams) -> pd.DataFrame:
    """
    Функция читает и производит минимальную предобработку датасета

    :param training_pipeline_params: схема общего конфига
    :return: обработанный pandas dataframe
    """

    if data is None:
        data = read_data(training_pipeline_params.input_data_path)
    data = drop_duplicates(data)
    data = fill_na(data)
    data = drop_useless_features(data, training_pipeline_params.feature_params)

    return data

def train_pipeline(data: pd.DataFrame, training_pipeline_params: TrainingPipelineParams) -> None:
    """
    Пайплайн-функция обучения модели

    :param data: pandas dataframe для обучения
    :param training_pipeline_params: схема общего конфига
    :return: None
    """
    logger.debug(f"Start train pipeline with params {training_pipeline_params}")
    logger.debug(f"data:  {data.shape} \n {data.info()} \n {data.nunique()}")

    # разбитие данных на выборки
    features, target = split_features_target(data, training_pipeline_params.feature_params)
    logger.debug(f"Split dataframe - features: {features.shape}, target: {target.shape}")

    features_train, features_test, target_train, target_test = split_train_test_data(
        features, target, training_pipeline_params.splitting_params
    )

    logger.debug("Split train and valid sample:")
    logger.debug(f"""features_train: {features_train.shape}, features_valid: {features_test.shape},
                 target_train: {target_train.shape}, target_valid: {target_test.shape}""")

    # обучение модели
    model = train_model(
            features_train, 
            target_train, 
            training_pipeline_params.train_params
        )

    # получение предсказаний и подсчет метрик
    preds = predict_model(model, features_test)
    metrics = evaluate_model(preds, target_test)
    logger.debug(f"preds/ targets shapes:  {(preds.shape, target_test.shape)}")

    # сохрание метрик в JSON
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f'Metric is {metrics}')

    # сохранение модели
    serialize_model(model, training_pipeline_params.output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    training_pipeline_params = create_schema_params(args.config)
    logger.info(f'удаляемые фичи {training_pipeline_params.feature_params.useless_features}')
    data = data_preprocessing(data=None, training_pipeline_params=training_pipeline_params)
    train_pipeline(data, training_pipeline_params)
