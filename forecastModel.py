import os
import warnings

import matplotlib
matplotlib.use('TkAgg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback


class CustomLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            loss = logs.get('loss', 0)
            mae = logs.get('mae', 0)
            print(f"Эпоха {epoch + 1}: loss={loss:.4f}, mae={mae:.4f}")


class WeatherForecastModel:
    def __init__(self, engine, model_dir='./models/', diagrams_dir='./files/diagrams/', window_size=12):
        """
        Конструктор, где задаём:
        - engine: подключение к БД (SQLAlchemy),
        - model_dir: путь для сохранения моделей,
        - diagrams_dir: путь для сохранения графиков,
        - window_size: "длина окна" (месяцев), на котором обучается LSTM.
        """
        self.engine = engine
        self.model_dir = model_dir
        self.diagrams_dir = diagrams_dir
        self.window_size = window_size
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.diagrams_dir, exist_ok=True)
        self.model = None

    def create_model(self):
        print("ℹ️ Создаём новую модель LSTM...")
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        self.model = model
        print("✅ Модель создана и скомпилирована.")

    def prepare_data(self, series):
        """
        Преобразование одномерного ряда (series) в набор (X, y),
        где X - "окна" длины window_size, y - следующий элемент ряда.
        """
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i+self.window_size])
            y.append(series[i+self.window_size])
        X = np.array(X).reshape((len(X), self.window_size, 1))
        y = np.array(y)
        return X, y

    def train_model(self, series, epochs=200, batch_size=8):
        print("ℹ️ Начинаем обучение модели...")
        X, y = self.prepare_data(series)
        if self.model is None:
            self.create_model()

        logging_callback = CustomLoggingCallback()

        # Запускаем обучение
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[logging_callback]
        )
        print("✅ Обучение завершено.")
        return history

    def evaluate_model(self, X, y):
        """
        Оцениваем точность (MAE) на переданных данных X,y.
        """
        loss, mae = self.model.evaluate(X, y, verbose=0)
        return mae

    def forecast(self, last_window, horizon=12):
        """
        Прогноз на 'horizon' шагов вперёд (итеративно).
        last_window - массив размером (window_size,),
        в начале - последние известные точки,
        на каждом шаге добавляем предсказанное значение и сдвигаем окно.
        """
        print(f"ℹ️ Делаем прогноз на {horizon} шагов вперёд...")
        predictions = []
        current_window = last_window.copy()
        for _ in range(horizon):
            input_data = current_window.reshape((1, self.window_size, 1))
            pred = self.model.predict(input_data, verbose=0)[0, 0]
            predictions.append(pred)
            current_window = np.append(current_window[1:], pred)
        print("✅ Прогноз рассчитан.")
        return predictions

    def forecast_with_dates(self, last_window, last_date, horizon=12):
        preds = self.forecast(last_window, horizon)
        dates = []
        d = pd.to_datetime(last_date)
        for _ in range(horizon):
            d += pd.DateOffset(months=1)
            dates.append((d.year, d.month))
        df = pd.DataFrame(dates, columns=['year', 'month'])
        df['predicted_temp'] = preds
        return df

    def plot_test_forecast(self, real_values, predicted_values, model_name):
        """
        График сравнения на тесте (два ряда: реальные и предсказанные)
        """
        print("ℹ️ Строим график сравнения на тестовой выборке...")
        plt.figure(figsize=(6, 4))
        plt.plot(real_values, marker='o', label='Реальные')
        plt.plot(predicted_values, marker='o', label='Предсказанные')
        plt.title('Сравнение на тестовой выборке (12 мес)')
        plt.ylabel('Температура')
        plt.legend()
        plt.xticks([])

        plot_path = os.path.join(self.diagrams_dir, f"test_forecast_{model_name}.png")
        plt.savefig(plot_path)
        print(f"✅ График сравнения на тестовой выборке сохранён в {plot_path}")
        plt.show()
        plt.close()
        return plot_path

    def plot_forecast(self, forecast_df, model_name):
        """
        График финального прогноза на будущее
        """
        print("ℹ️ Строим график прогноза на будущие 12 месяцев...")
        plt.figure(figsize=(6, 4))
        plt.plot(forecast_df['predicted_temp'], marker='o')
        plt.title('Прогноз на 12 месяцев вперёд')
        plt.ylabel('Температура')
        plt.xticks([])

        plot_path = os.path.join(self.diagrams_dir, f"forecast_{model_name}.png")
        plt.savefig(plot_path)
        print(f"✅ График прогноза сохранён в {plot_path}")
        plt.show()
        plt.close()
        return plot_path

    def plot_training_loss(self, history, model_name):
        """
        График потерь (loss=MAE) во время обучения.
        """
        print("ℹ️ Строим график потерь финального обучения...")
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Потери (loss)')
        plt.title('График потерь во время обучения')
        plt.xlabel('Эпоха')
        plt.ylabel('Loss (MAE)')

        plot_path = os.path.join(self.diagrams_dir, f"loss_{model_name}.png")
        plt.savefig(plot_path)
        print(f"✅ График потерь сохранён в {plot_path}")
        plt.show()
        plt.close()
        return plot_path

    def save_model(self, model_name, last_date, test_mae,
                   test_forecast_plot, loss_plot, forecast_plot):

        print("ℹ️ Сохраняем модель и информацию о ней в БД...")
        try:
            path = os.path.join(self.model_dir, model_name)
            self.model.save(path)

            # Создаём таблицу models, если не существует
            with self.engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        last_date TEXT,
                        test_mae REAL,
                        test_forecast_plot TEXT,
                        loss_plot TEXT,
                        forecast_plot TEXT
                    )
                """))
                query = text("""
                    INSERT INTO models (
                        model_name,
                        last_date,
                        test_mae,
                        test_forecast_plot,
                        loss_plot,
                        forecast_plot
                    ) VALUES (
                        :model_name,
                        :last_date,
                        :test_mae,
                        :test_forecast_plot,
                        :loss_plot,
                        :forecast_plot
                    )
                """)
                conn.execute(query, {
                    "model_name": model_name,
                    "last_date": last_date,
                    "test_mae": test_mae,
                    "test_forecast_plot": test_forecast_plot,
                    "loss_plot": loss_plot,
                    "forecast_plot": forecast_plot
                })

            print(f"✅ Модель '{model_name}' сохранена вместе с 3 графиками в БД.")
        except Exception as e:
            print(f"❌ Ошибка при сохранении модели: {e}")

    def load_model(self, model_path):
        """
        Загрузка уже обученной модели с диска (файл .keras).
        """
        print(f"ℹ️ Загружаем модель из файла: {model_path}")
        self.model = load_model(model_path)
        print("✅ Модель успешно загружена.")
