# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.entities.split_params import SplittingParams
from src.entities.feature_params import FeatureParams


def read_data(path: str) -> pd.DataFrame:
    """
    Читает CSV датасет и возвращает Pandas dataframe

    :param path: путь к датасету
    :return: объект датафрейма Pandas
    """
    return pd.read_csv(path)


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет полные дубликаты строк в датафрейме

    :param data: pandas датафрейм с дубликатами
    :return: pandas датафрейм без дубликатов
    """
    return data.drop_duplicates().reset_index(drop=True)


def fill_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция заполняет пропуски медианой в признаках

    :param data: pandas датафрейм
    :return: pandas датафрейм без пропусков
    """
    na_series = data.isna().sum()

    if len(na_series[na_series != 0]) == 0:
        return data
    
    columns_with_na = na_series[na_series != 0].index
    data[columns_with_na] = data[columns_with_na].fillna(data[columns_with_na].median())

    return data

def drop_useless_features(data: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    """
    Функция удаляет из датасета малополезные признаки после анализа через Boruta

    :param data: pandas датафрейм
    :param params: датакласс конфига фич
    :return: pandas датафрейм без указанных столбцов
    """
    return data.drop(params.useless_features, axis=1)
    


def split_features_target(data: pd.DataFrame, params: FeatureParams
                          ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Функция разбивает датасет на признаки и целевой признак

    :param data: pandas датафрейм
    :param params: датакласс конфига фич
    :return: кортеж признаков и таргета
    """

    features = data.drop(params.target_col, axis=1)
    target = data[params.target_col]

    return features, target


def split_train_test_data(
        features: pd.DataFrame, 
        target: pd.Series, 
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Функция разбивает признаки и таргет на трейн и тест

    :param features: pandas датафрейм с признаками
    :param target: pandas series с таргетом
    :param params: датакласс конфига фич
    :return: кортеж выборок трейна и теста
    """

    features_train, features_test, target_train, target_test = train_test_split(
        features, 
        target,
        test_size=params.val_size, 
        random_state=params.random_state, 
        shuffle=params.shuffle,
        stratify=target
    )

    return features_train, features_test, target_train, target_test