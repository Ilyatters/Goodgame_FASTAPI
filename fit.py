import json
import pandas as pd
import config
# Классы реализации модели
# from ModelTfrs import ModelTfrs
from classes.ModelFastai import ModelFastai
# from ModelLightFM import ModelLightFM
# from ModelGMF import ModelGMF
# from ModelNCF import ModelNCF
# from ModelCollaborativeF import ModelCollaborativeF
# from NNHS import NNHS


def fit(CORE):
    print('--- FIT MODEL ---')

    days = 1 if 'days' not in CORE.req else CORE.req['days']
    mn = 1 if 'model' not in CORE.req else int(CORE.req['model'])

    # Добавляем модели в список
    # model_list = [ModelTfrs, ModelFastai, ModelLightFM, ModelGMF, ModelNCF, ModelCollaborativeF, NNHS]
    model_list = [ModelFastai]

    if mn < 1 or mn > len(model_list):
        mn = 1
    CORE.model = model_list[mn-1](CORE.config)

    # Получение датасета duration 
    # days = 1  # Для теста

    answer = CORE.model.fit(days)  # Получаем данные по API
    return answer
