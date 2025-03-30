import pandas as pd
import json
import os
from classes.ModelFastai import ModelFastai
from config import config

model = ModelFastai(config)

result_fit = model.fit()

# Проверяем результат обучения
if result_fit['status'] == 'OK':
    print("Модель успешно обучена и сохранена.")
else:
    print("Ошибка при обучении модели:", result_fit['message'])

# 3. Проверка данных
print(f"Загружено {len(model.df_duration)} записей")
print(f"Уникальных пользователей: {model.df_final['user_id'].nunique()}")