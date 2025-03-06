from sqlalchemy import create_engine
import pandas as pd

class ProcessingData:
    def __init__(self):
        # Создаем подключение к базе данных SQLite
        self.engine = create_engine('sqlite:///trajectory.db')

    def parseData(self, path: str):
        # Чтение данных из Excel
        df = pd.read_excel(path)

        # Выбираем только нужные столбцы
        df = df.iloc[:, [0, 1]]

        # Переименовываем столбцы
        df.columns.values[0] = 'date'
        df.columns.values[1] = 'temp'

        # Преобразуем столбец 'date' в datetime формат
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')

        # Добавляем столбцы с годом и месяцем
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Заполняем пропущенные значения температуры средним по месяцу
        df['temp'] = df.groupby('month')['temp'].transform(lambda x: x.fillna(x.mean()))

        # Заполняем пропущенные значения для 'year' и 'month', если они есть
        df['year'] = df['year'].fillna(df['year'].mode()[0]).astype(int)
        df['month'] = df['month'].fillna(df['month'].mode()[0]).astype(int)

        # Группируем по году и месяцу и считаем среднюю температуру
        monthly_avg_temp = df.groupby(['year', 'month'])['temp'].mean().reset_index()

        # Сохраняем данные в базу данных
        monthly_avg_temp.to_sql('monthly_weather_data', con=self.engine, if_exists='replace', index=False)
