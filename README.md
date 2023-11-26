# car_price_predictor
This repository contains #1 homework  within the framework of HSE ML course

## Что было сделано

В рамках данного задания:
* был проведен разведочный анализ данных и минимальная обработка признаков в том числе конвертация строковых столбцов в числовые, заполнение пропусков медианами, а также были построены графики для визуального анализа данных (`heatmap`, `pairplot` и тд.)
* были простроены линейные модели `Ridge`, `Lasso`, `Elastic`, а также обычная линейная регрессия без регуляризации на числовых признаках и с добавление категориальных признаков, которые были закодированы с помощью `OneHotEncoder`. Для подбора гиперпараметров моделей с регуляризацией был использован `GridSearchCV` с 10-ю фолдами. Что не менее важно, была применена стандатизация признаков с помощью `StandardScaler`.
* был реализован сервис, который предсказывает стоимость автомобиле. В нем используется лучшая модель с $R^2 = 0.62$ на тестовой выборке. Параметры, скейлеры и энкодеры сохранены в формате `pickle` для оптимизации предсказаний. Доступ к сервису осуществляется посредством API `predict_item`, `predict_items` реализованного с помощью `FastAPI`

## Результаты и ретроспектива
Результатом данной домашней работы является сервис, который осуществляет предсказания стоимоисти автомобилей. Наибольший буст в качестве предсказаний дало добавление категориальных признаков и стандартизация.

## Что не получилось или что не было сделано
Не были реализованы бонусные части:
* Не был проведен анализ выбросов
* Не был проведен feature engineering

Также не был оптимизирован сервис. Он реализован довольно примитивно

Нет особых причин почему данные части упражнения не были сделаны: просто, я подумал, что уже достаточно.