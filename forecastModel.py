import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class WeatherForecastModel:
    def __init__(self, engine, model_dir='./models/', window_size=12):
        """
        :param engine: SQLAlchemy engine для работы с БД
        :param model_dir: директория для сохранения модели
        :param window_size: размер окна (количество предыдущих месяцев, используемых для предсказания)
        """
        self.engine = engine
        self.model_dir = model_dir
        self.window_size = window_size
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None

    def create_model(self):
        """Создание модели на основе LSTM для одномерного временного ряда."""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        self.model = model
        return model

    def prepare_data(self, series):
        """
        Готовит данные для обучения.
        :param series: одномерный numpy-массив (например, температура)
        :return: кортеж (X, y), где X имеет форму (samples, window_size, 1)
        """
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i+self.window_size])
            y.append(series[i+self.window_size])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], self.window_size, 1))
        return X, y

    def train_model(self, series, epochs=50, batch_size=16):
        """
        Обучает модель на временном ряде.
        :param series: одномерный numpy-массив (например, среднемесячная температура)
        :param epochs: число эпох обучения
        :param batch_size: размер батча
        :return: история обучения
        """
        X, y = self.prepare_data(series)
        if self.model is None:
            self.create_model()
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def save_model(self, model_filename):
        # Сохраняем модель в файл (используем формат .keras)
        self.model.save(os.path.join(self.model_dir, model_filename))

        # Создаем таблицу моделей, если она еще не существует
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL
                )
            """))

            # Добавляем название модели в таблицу
            query = text("INSERT INTO models (model_name) VALUES (:model_name)")
            conn.execute(query, {"model_name": model_filename})

        print(f"Модель сохранена как {model_filename} и добавлена в базу данных.")

    def forecast(self, last_window, forecast_horizon=12):
        """
        Получает прогноз для следующих forecast_horizon периодов.
        :param last_window: numpy-массив длины window_size с последними известными значениями
        :param forecast_horizon: число периодов для прогнозирования (по умолчанию 12 месяцев)
        :return: список предсказанных значений
        """
        predictions = []
        current_window = last_window.copy()
        for _ in range(forecast_horizon):
            # Меняем форму входного массива на (1, window_size, 1)
            input_data = current_window.reshape((1, self.window_size, 1))
            pred = self.model.predict(input_data, verbose=0)
            predictions.append(pred[0, 0])
            # Обновляем окно: убираем первый элемент и добавляем предсказанный
            current_window = np.append(current_window[1:], pred[0, 0])
        return predictions

    def forecast_with_dates(self, last_window, last_date, forecast_horizon=12):
        """
        Получает прогноз на следующие 12 месяцев и возвращает DataFrame с годом и месяцем для каждого предсказания.
        :param last_window: numpy-массив с последними window_size значениями (например, температуры)
        :param last_date: последняя дата наблюдения (строка или datetime, например, '2024-12-01')
        :param forecast_horizon: число месяцев для прогнозирования
        :return: DataFrame с колонками ['year', 'month', 'predicted_temp']
        """
        predictions = self.forecast(last_window, forecast_horizon)
        forecast_dates = []
        current_date = pd.to_datetime(last_date)
        for _ in range(forecast_horizon):
            current_date = current_date + pd.DateOffset(months=1)
            forecast_dates.append((current_date.year, current_date.month))
        df_forecast = pd.DataFrame(forecast_dates, columns=['year', 'month'])
        df_forecast['predicted_temp'] = predictions
        return df_forecast
