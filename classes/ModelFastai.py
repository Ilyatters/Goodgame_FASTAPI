import os
import sys
import logging
import time
import datetime
import warnings
from datetime import datetime
from attr import validate
import requests
import pandas as pd
import numpy as np
from typing import Dict, Text
from pymongo import MongoClient
from pathlib import Path

import torch, fastai
from fastai.collab import *
from fastai.tabular.all import *
from fastai.learner import load_learner

from recommenders.models.fastai.fastai_utils import cartesian_product

# print("System version: {}".format(sys.version))
# print("Pandas version: {}".format(pd.__version__))
# print("Fast AI version: {}".format(fastai.__version__))
# print("Torch version: {}".format(torch.__version__))
# print("Cuda Available: {}".format(torch.cuda.is_available()))
# print("CuDNN Enabled: {}".format(torch.backends.cudnn.enabled))

class ModelFastai:
    def __init__(self, config):
        print('-----MODEL INIT-----')
        self.config = config  # Данные из конфигурационного файла
        self.data_duration = []  # Временное хранилище списка получаемых данных
        self.df_duration = None  # Тут размещаем данные 'duration' в формате Pandas, полученные результате работы метода __get_data_duration
        self.df_final = None # Пост-обработанные данные 'duration' в формате Pandas, полученные результате работы метода __data_processing
        self.df_website = None  # Тут размещаем данные 'website' в формате Pandas полученные результате работы метода __get_data_website
        self.log = [] # Лог операций
        self.model = None  # Тут будет храниться модель
        self.run_time = 0
        
        self.files_path = Path(config.path).resolve() / 'ModelFastai'
        self.files_path.mkdir(parents=True, exist_ok=True, mode=0o755)
        
        # Пример правильного формирования путей:
        self.model_path = self.files_path / 'model.pkl'
        self.log_path = self.files_path / 'status.log'
        self.predictions_path = self.files_path / 'users_items_predicts.csv'

        # Создаём папку для файлов модели
        if not os.path.isdir(self.files_path):
            os.mkdir(self.files_path, mode=0o755)

    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def fit(self, days=30):
        # Удаляем файл логов
        if self.log_path.exists():
            self.log_path.unlink() 

        self.run_time = 0  # Обнуляем время выполнения

        # Получаем данные `duration` за указанное количество дней `days`
        # answer = self.__get_data_duration(days)
        # if answer['status'] != 'OK':
            # return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        answer = self.data_csv()
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        # Получаем данные `website` за указанное количество дней `days`
        # self.__get_data_website(days) 

        # Обработка данных
        answer = self.__data_processing()
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        # Обучение модели
        answer = self.fit_model()
        if answer['status'] != 'OK':
            return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        # Расчет рекомендаций
        # answer = self.compute_predicts()
        # if answer['status'] != 'OK':
           # return answer  # Возвращаем ответ, если возникла ошибка, прерываем дальнейшее выполнение

        return answer

    # Предсказание - переопределяем в наследуемом классе
    def predict(self, user_id):
        print('-----MODEL PREDICT-----')
        start_time = time.time()

        model_path = self.files_path / 'model.pkl'
        print("Model path:", model_path)  # Убедитесь, что путь правильный
        self.model = load_learner(model_path)

        n = 10 # self.config.top # сколько каналов рекомендуем 
        # Get all users and items that the model knows /content/корень/ModelFastai/model.pkl

        _, total_items = self.model.classes.values()
        total_items = total_items[1:]
        total_users = [str(user_id)]

        users_items = cartesian_product(np.array(total_users),np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=['user_id','channel_id'])

        dl_test = self.model.dls.test_dl(users_items, with_labels=True)
        preds = self.model.get_preds(dl=dl_test, with_decoded=True)
        users_items['preds'] = preds[2]
        users_items.sort_values(by='preds', ascending=False).head(10)

        delta_time = time.time() - start_time

        print(f'Predict execution time: {round(delta_time, 4)}s')

        channel_list=users_items[users_items["user_id"]==str(user_id)].sort_values(by='preds', ascending=False).head(n)

        return channel_list['channel_id'].tolist()


    # === ПОЛУЧЕНИЕ ДАННЫХ "DURATION" ПО API ЗА НУЖНОЕ КОЛИЧЕСТВО ДНЕЙ ===
    # def __get_data_duration(self,  days=30):
        print('-----ЗАГРУЖАЕМ ДАННЫЕ-----')

        start_time = time.time()
        from_time = int(time.time()) - int(days) * 86400
        to_time = int(time.time())

    # Параметры запроса
        params = {
            "from": from_time,
            "to": to_time,
            "key": self.config.model_key  # API Key из конфига
        }
        api_req = requests.get(self.config.gg_api_url, params=params)
        print('---Ссылка на API---')
        print(api_req.request.url)
        # --- Соединение с GG Api и получение данных ---
        for i in range(1, 10):
            # url = self.config.gg_api_url + from_time_url + from_to_time + '&page=' + str(i)
            # url = f"{self.config.gg_api_url}?from={from_time}&to={to_time}&page={i}"

            try:
                if api_req.status_code != 200:
                    answer = {
                        'status': 'ERROR', 
                        'message': f'Error connecting to goodgame.ru API. Server API response code: {api_req.status_code}'
                    }
                    self.__logging(answer)
                    break

                else:
                    # Добавляем данные в наш список
                    data_list = api_req.json()
                    self.data_duration.extend(data_list)
                    self.df_duration = pd.DataFrame.from_dict(self.data_duration)
                    self.df_duration = self.df_duration.drop(columns=['streamer_id', 'user'])  # Удаляем колонку

                    if len(data_list) < self.config.gg_pagination_step:  # Шаг пагинации
                        print(f"Total received records: {len(self.data_duration)}")
                        # self.df_duration = pd.DataFrame.from_dict(self.data_duration)
                        break
            except Exception as e:
                    answer = {
                    'status': 'ERROR', 
                    'message': f'Error connecting to goodgame.ru API: {str(e)}'
                     }
                    self.__logging(answer)
                    break

        # --- Формирование ответа и запись в лог ---
        delta_time = time.time() - start_time
        if (len(self.data_duration) > 0):
            answer = {
                'status': 'OK', 
                'message': f'Data received, number of rows: {len(self.data_duration)}, execution time: {round(delta_time, 4)}s'
            }
            self.__logging(answer)
        else:
            answer = {
                'status': 'ERROR', 
                'message': f'No data, execution time: {round(delta_time, 4)}s'
            }
            self.__logging(answer)
        self.data_duration = []  # Очищаем буфер данных

        self.run_time += delta_time

        return answer 

    def data_csv(self):
        print('-----ЗАГРУЖАЕМ ДАННЫЕ-----')
        start_time = time.time()

        try:
            # Читаем данные из CSV файла
            self.df_duration = pd.read_csv('channels_big_data.csv')

            self.df_duration = self.df_duration.copy()
            self.df_duration = self.df_duration.drop(columns=['streamer_id', 'user'])  # Удаляем колонку

            # Преобразуем столбец 'time' в формат datetime для работы с датой и временем
            self.df_duration['time'] = pd.to_datetime(self.df_duration['time'])

            #start_date = pd.to_datetime('2025-02-20')
            #end_date = pd.to_datetime('2025-03-22')
            
            #self.df_duration = self.df_duration[(self.df_duration['time'] >= start_date) & (self.df_duration['time'] <= end_date)]
            print('-' * 100)
            print('Вывод первых и последних строк загруженной таблицы для наглядности')
            print(self.df_duration.head(-5))


            delta_time = time.time() - start_time
            answer = {
                'status': 'OK', 
                'message': f'Data received, number of rows: {len(self.df_duration)}, execution time: {round(delta_time, 4)}s'
            }
        except Exception as e:
            delta_time = time.time() - start_time
            answer = {
                'status': 'ERROR', 
                'message': f'Error loading data: {str(e)}, execution time: {round(delta_time, 4)}s'
            }

        self.__logging(answer)
        self.run_time += delta_time
        
        return answer 

    # === ОБРАБОТКА ДАННЫХ ===
    def __data_processing(self):
        print('-----ОБРАБОТКА ДАННЫХ-----')
        start_time = time.time()
        
        if self.df_duration is None:
            answer = {
                'status': 'ERROR', 
                'message': 'Данные не были получены. Проверьте метод __get_data_duration.'
            }
            self.__logging(answer)
            return answer

        df = self.df_duration
        df.dropna(inplace=True)

        if df.empty:
            answer = {
                'status': 'ERROR', 
                'message': 'DataFrame пустой. Проверь, что данные были успешно получены.'
            }
            self.__logging(answer)
            return answer

        if 'channel_id' not in df.columns:
            answer = {
                'status': 'ERROR', 
                'message': f"Столбец 'channel_id' отсутствует в DataFrame. Доступные столбцы: {df.columns.tolist()}"
            }
            self.__logging(answer)
            return answer
    
        # --- Находим самые популярные каналы - составляем список из 100 каналов ---
        self.popular_list = df.groupby(['channel_id']).size().sort_values(ascending=False)[0:300].tolist()

        print('-' * 100)
        print("Находим самые популярные каналы - составляем список из 100 каналов")
        print(self.popular_list[:20], '......')
        print('-' * 100)
        print(f"Нашли топ {len(self.popular_list)} каналов") 

        # --- Удаляем стримы с небольшим количество просмотров (20% от среднего) ---
        # Находим среднее число просмотров стрима
        view_mean = df.groupby(['channel_id']).size().mean()
        print('-' * 100)
        print(f"Среднее число по количеству просмотров для одного стрима: {view_mean}")

        # Создадим новый датасет только с подсчетом количества просмотров, чтобы исключить некоторые
        view_counter = pd.DataFrame({'Count' : df.groupby(['channel_id']).size()}).reset_index()
        print('-' * 100)
        print("Таблица с подсчетом количества просмотров для каждого канала")
        print(view_counter.head(-3))

        # Убираем каналы, с наименьшим количеством просмотров. Меньше двух просмотров
        view_counter = view_counter.loc[view_counter['Count'] >= 2]

        print('-' * 100)
        print("Измененная таблица после удаления слабых каналов")
        print(view_counter.head(-3))

        # Оставляем самые просматриваемые каналы и формируем измененную таблицу
        reducer = df['channel_id'].isin(view_counter['channel_id'])
        df_2 = df[reducer]
        self.df_final = df_2.drop_duplicates()
        print('-' * 100)
        print('Отсортированная таблица')
        print(self.df_final.head(-3))

        print('-' * 100)
        print('Суммируем данные каждого канала для составления рейтинга')
        # Сначала суммируем только duration
        sum_duration = self.df_final.groupby(['user_id', 'channel_id'])['duration'].sum().reset_index()
        # Затем добавляем другие нечисловые колонки 
        df_time = df.groupby(['user_id', 'channel_id'])['time'].first().reset_index()
        self.df_final = pd.merge(sum_duration, df_time, on=['user_id', 'channel_id'])
        print('-' * 100)
        print('Таблица перед составлением рейтинга')
        print(self.df_final.head(-3))

        # Проверка данных
        print('-' * 100)
        print(f"Загружено {len(self.df_final)} записей")
        print(f"Уникальных пользователей: {self.df_final['user_id'].nunique()}")

        #Образуем систему рейтинга 
        self.df_final['duration'] = np.log1p(self.df_final['duration'])  # Логарифмическое преобразование   
        self.df_final['duration'] = self.df_final['duration'].clip(upper=90*1000*60)
        self.df_final['rating'] = pd.qcut(self.df_final['duration'], q=5, labels=[1.0, 2.0, 3.0, 4.0, 5.0])

        #all_users = self.df_final['user_id'].unique()
        #all_channels = self.df_final['channel_id'].unique()
        #all_pairs = pd.DataFrame(cartesian_product(all_users, all_channels), columns=['user_id', 'channel_id'])
        #all_pairs['rating'] = 0.0
        #self.df_final = pd.concat([self.df_final, all_pairs]).drop_duplicates(subset=['user_id', 'channel_id'], keep='first')

        print('-' * 100)
        print(f'Количество каналов с рейтингом 5 = {len(self.df_final[self.df_final["rating"]==5.0])}')
        print(f'Количество каналов с рейтингом 4 = {len(self.df_final[self.df_final["rating"]==4.0])}')
        print(f'Количество каналов с рейтингом 3 = {len(self.df_final[self.df_final["rating"]==3.0])}')
        print(f'Количество каналов с рейтингом 2 = {len(self.df_final[self.df_final["rating"]==2.0])}')
        print(f'Количество каналов с рейтингом 1 = {len(self.df_final[self.df_final["rating"]==1.0])}')

        # чистка лишних столбцов
        self.df_final = self.df_final.drop(columns=['time', 'duration'], errors='ignore')
        print('-' * 100)
        print('Итоговая таблица для обучения')
        print(self.df_final.head(-3))

        delta_time = time.time() - start_time
        self.run_time += delta_time

        answer = {
            'status': 'OK', 
            'message': f'Data processed, number of rows: {self.df_final.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)
        return answer


    # === ЗАПИСЬ В ЛОГ ===
    def __logging(self, answer):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            log_entry = f"{datetime.now()}, {answer['status']}: {answer['message']}\n"
            f.write(log_entry)



    # === ОБУЧЕНИЕ МОДЕЛИ ===
    def fit_model(self):
        print('-----MODEL FIT-----')
        start_time = time.time()

        dls = CollabDataLoaders.from_df(self.df_final, validate_pct=0.3)

        #embs = get_emb_sz(dls)
#
        #model = DotProductBias(n_users, n_movies, 50)
        #learn = Learner(dls, model, loss_func=MSELossFlat())
        #learn.fit_one_cycle(5, 5e-3, wd=0.1)

        # инициализация модели
        self.model = collab_learner(dls, n_factors=128, y_range=[0, 5], wd=0.3)
        #self.model = collab_learner(dls, use_nn=True, layers=[32, 12], y_range=[0.5, 5.5], wd=0.2)
        print(self.model.model) # справка по модели

        # старт обучения
        self.model.fit_one_cycle(10, 5e-3, wd=0.3)

        # экспорт модели 
        self.model.export(self.files_path / 'model.pkl')

        delta_time = time.time() - start_time
        self.run_time += delta_time

        answer = {
            'status': 'OK', 
            'message': f"Took {round(self.run_time)} seconds for training."
        }
        self.__logging(answer)

        return answer

    def cartesian_product(*arrays):
         """Compute the Cartesian product in fastai algo. This is a helper function.

         Args:
             arrays (tuple of numpy.ndarray): Input arrays

         Returns:
             numpy.ndarray: product

         """
         la = len(arrays)
         dtype = np.result_type(*arrays)
         arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
         for i, a in enumerate(np.ix_(*arrays)):
             arr[..., i] = a
         return arr.reshape(-1, la)

    def compute_predicts(self):
        try:
            # 1. Проверка существования файла модели
            model_path = Path(self.files_path) / 'model.pkl'
            if not model_path.is_file():
                return {
                    'status': 'ERROR',
                    'message': f'Файл модели не найден по пути: {model_path}'
                }

            # 2. Проверка наличия обработанных данных
            if self.df_final is None or self.df_final.empty:
                return {
                    'status': 'ERROR',
                    'message': 'Данные не были обработаны. Сначала выполните fit()'
                }

            # 3. Загрузка модели с обработкой предупреждений
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.model = load_learner(model_path)
                except Exception as e:
                    return {
                        'status': 'ERROR',
                        'message': f'Ошибка загрузки модели: {str(e)}'
                    }

            # 4. Проверка загруженной модели
            if not hasattr(self.model, 'dls') or self.model.dls is None:
                return {
                    'status': 'ERROR',
                    'message': 'Модель загружена некорректно'
                }

            # 5. Получение списка каналов
            try:
                _, total_items = self.model.dls.classes.values()
                total_items = total_items[1:]  # Пропускаем первый элемент
            except Exception as e:
                return {
                    'status': 'ERROR',
                    'message': f'Ошибка получения списка каналов: {str(e)}'
                }

            # 6. Получение списка пользователей
            try:
                total_users = list(self.df_final['user_id'].astype(str).unique())
            except Exception as e:
                return {
                    'status': 'ERROR',
                    'message': f'Ошибка получения списка пользователей: {str(e)}'
                }

            # 7. Создание комбинаций и предсказание
            try:
                users_items = cartesian_product(np.array(total_users), np.array(total_items))
                users_items = pd.DataFrame(users_items, columns=['userID', 'itemID'])
                dl_test = self.model.dls.test_dl(users_items, with_labels=True)
                preds = self.model.get_preds(dl=dl_test, with_decoded=True)
                users_items['prediction'] = preds[2]
            except Exception as e:
                return {
                    'status': 'ERROR',
                    'message': f'Ошибка при создании предсказаний: {str(e)}'
                }

            # 8. Сохранение результатов
            try:
                output_path = Path(self.files_path) / 'users_items_predicts.csv'
                users_items.to_csv(output_path, index=False)
                return {
                    'status': 'OK',
                    'message': f'Предсказания успешно сохранены в {output_path}',
                    'predictions': users_items.head().to_dict('records')  # Пример данных
                }
            except Exception as e:
                return {
                    'status': 'ERROR',
                    'message': f'Ошибка при сохранении результатов: {str(e)}'
                }

        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Неожиданная ошибка: {str(e)}'
            }
