import pandas as pd
import json
import os
from classes.ModelFastai import ModelFastai
from config import config

model = ModelFastai(config)

fit_result = model.fit(days=30)
if fit_result['status'] != 'OK':
    print(f"Ошибка обучения: {fit_result['message']}")
    exit()

# 3. Проверка данных
print(f"Загружено {len(model.df_final)} записей")
print(f"Уникальных пользователей: {model.df_final['userID'].nunique()}")

# 4. Вычисление предсказаний
predict_result = model.compute_predicts()
if predict_result['status'] == 'OK':
    print(predict_result['message'])
    print("Пример предсказаний:", predict_result.get('predictions', []))
else:
    print(f"Ошибка: {predict_result['message']}")