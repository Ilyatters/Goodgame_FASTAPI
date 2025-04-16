import pandas as pd
import numpy as np
from classes.ModelFastai import ModelFastai
from config import config

# Задаем пользователя для проверки
user_id = 1621033

# Определяем свою модель, передаем все необходимые параметры (опционально)
model = ModelFastai(config)

# Задаем первый метод для обучения модели (ничего не передаем, т.к. каждый обучал свою модель через csv-файл)
result_fit = model.fit()

# Задаем второй метод для предсказания модели и передаем user_id
result_predict_channel = model.predict(user_id=user_id)


# Проверяем результат обучения
if result_fit['status'] == 'OK':
    print('-' * 100)
    print("Модель успешно обучена и сохранена.")
else:
    print('-' * 100)
    print("Ошибка при обучении модели:", result_fit['message'])

# Проверяем, что это не пустой список
if isinstance(result_predict_channel, list) and len(result_predict_channel) > 0:
    print('-' * 100)  
    print(f"Рекомендованные каналы для пользователя '{user_id}': {result_predict_channel}")
else:
    print('-' * 100)
    print("Ошибка: нет данных")


# Cоздаем датасет для проверки реальной истории пользователя. Из основной директории подгружаем заранее скачанный csv-файл
data = 'data.csv'
df_csv = pd.read_csv(data)
df_csv = df_csv.copy()

# Удаляем (пока что) не нужные столбцы
df = df_csv.drop(columns= ['streamer_id', 'user'])

# Преобразуем столбец 'time' в формат datetime для работы с датой и временем
df['time'] = pd.to_datetime(df['time'])

# Задаем временные рамки и образуем таблицу с историей всех пользователей за 5 дней
start_date = pd.to_datetime('2025-03-22')
end_date = pd.to_datetime('2025-03-27')
check_data = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

print('-' * 100)
print("Таблица с историческими данными для всех пользователей за 5 дней")
print(check_data.head(-3))

# Таблица с рейтингом для предсказанного пользователя
# Берем данные из таблицы для определенного user_id
user_data = check_data[check_data['user_id'] == user_id]
print('-' * 100)
print(f'Данные из таблицы для определенного user_id: {user_id}')
if user_data.empty:
    print(f'У пользователя "{user_id}" нет истории')
else:
    print(user_data.head(-3))
    # Группируем по duration
    top_channels = user_data.groupby('channel_id').agg({'duration':'sum'}).sort_values('duration', ascending=False)

    # Задаем метрику времени в минутах, для удобной оценки
    top_channels['minutes'] = (top_channels['duration'] / (1000 * 60)).astype(int)

    # Задаем рейтинг для каждого канала через duration
    top_channels['rating'] = np.where(top_channels['duration'] < 10*1000*60, 1.0,
                                      # первый множитель = колво минут просмотра.
                         np.where(top_channels['duration'] < 30*1000*60, 2.0,
                                      # через запятую = присваиваемый рейтинг.
                         np.where(top_channels['duration'] < 60*1000*60, 3.0,
                                      # через вторую запятую пишем, что будет при ложном условии
                         np.where(top_channels['duration'] < 120*1000*60,4.0,5.0))))
    print('-' * 100)
    print(f"Суммарное время просмотра для топ-10 каналов пользователя {user_id}: ")
    print(top_channels.head(10))

        # Делаем проверку целостности данных для успешного заполнения таблицы
    print('-' * 100)
    print("Проверяем одинаковую длину данных для каждого столбца")
    print(result_predict_channel, len(result_predict_channel))
    print(top_channels.index[:10].tolist(), len(top_channels.index[:10].tolist()))
    print(top_channels['duration'].head(10).tolist(), len(top_channels['duration'].head(10).tolist()))
    print(top_channels['minutes'].head(10).tolist(), len(top_channels['minutes'].head(10).tolist()))
    print(top_channels['rating'].head(10).tolist(), len(top_channels['rating'].head(10).tolist()))

    # Формируем данные для таблицы
    data = {
        'recom_channel' : result_predict_channel,
        'history top' : top_channels.index[:10].tolist(),
        'actual duration' : top_channels['duration'].head(10).tolist(),
        'actual min' : top_channels['minutes'].head(10).tolist()
    }

    result_df = pd.DataFrame(data)

    print('-' * 100)
    print(f'Итоговая таблица для сравнения предсказания и реальных действий user_id: {user_id}')
    print(result_df)
