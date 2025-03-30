from classes.Core import Core
from config import config
import fit # Импортируем модуль fit


# Инициализация
CORE = Core()
CORE.config = config
CORE.req = {
    'key': config.model_key,  # Ключ из config.py
    'act': 'fit',
    'days': 1,  # Количество дней данных
    'model': 1   # Выбор ModelFastai
}

# Запуск обучения
result = fit.fit(CORE)

# Вывод результатов
print("\n=== Результат обучения ===")
print(f"Статус: {result['status']}")
print(f"Сообщение: {result['message']}")