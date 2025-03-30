import pandas as pd
import json
import os
from classes.ModelFastai import ModelFastai
from config import config

model = ModelFastai(config)

result = model.predict(user_id=102)

if isinstance(result, list) and len(result) > 0:  # Проверяем, что это непустой список
    print("Рекомендованные каналы:", result)
else:
    print("Ошибка: нет данных")

