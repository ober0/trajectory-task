import pandas as pd
import time
import os
from datetime import datetime
from sqlalchemy import text
from processingData import ProcessingData
from forecastModel import WeatherForecastModel


def process_and_train_model(filename: str, input_model_name: str):
    """
    Обучаем новую модель, строим 3 графика (тест, прогноз, потери),
    сохраняем модель и итоговый прогноз в файл (Excel c датой в имени).
    Возвращает (путь к Excel, имя модели).
    """
    try:
        print("ℹ️ Идёт парсинг данных и подготовка...")
        p = ProcessingData()
        p.parseData(filename)

        data = p.get_weather_data()
        last_date = p.get_latest_date()

        data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
        data = data.sort_values(by='date')
        series = data['temp'].values

        if len(series) < 24:
            print("❌ Недостаточно данных (меньше 24).")
            return None, None

        # Разделяем: train / test
        train_part = series[:-12]
        test_part = series[-12:]

        # Имя модели (.keras)
        model_name = f"{input_model_name}_{int(time.time())}.keras"
        wf = WeatherForecastModel(p.engine, window_size=12)

        print("ℹ️ Обучаем модель на train (проверка на тесте)...")
        wf.create_model()
        wf.train_model(train_part, epochs=500, batch_size=8)

        print("ℹ️ Прогноз на тест (12 мес)...")
        last_window_train = train_part[-wf.window_size:]
        test_preds = wf.forecast(last_window_train, horizon=12)
        test_mae = abs(test_part - test_preds).mean()
        print(f"✅ MAE на тестовой выборке = {test_mae:.4f}")

        print("ℹ️ Строим график сравнения (тест)...")
        test_forecast_plot = wf.plot_test_forecast(test_part, test_preds, model_name)

        print("ℹ️ Обучаем модель на ПОЛНОМ датасете (train+test)...")
        wf.create_model()
        history_full = wf.train_model(series, epochs=500, batch_size=8)

        print("ℹ️ Прогноз на 12 месяцев вперёд...")
        last_window_full = series[-wf.window_size:]
        forecast_df = wf.forecast_with_dates(last_window_full, last_date, horizon=12)

        # График прогноза
        forecast_plot = wf.plot_forecast(forecast_df, model_name)

        # График потерь
        loss_plot = wf.plot_training_loss(history_full, model_name)

        # Сохраняем модель (3 графика) в БД
        wf.save_model(
            model_name=model_name,
            last_date=last_date,
            test_mae=float(test_mae),
            test_forecast_plot=test_forecast_plot,
            loss_plot=loss_plot,
            forecast_plot=forecast_plot
        )

        # Сохраняем финальный прогноз в Excel (с датой в названии)
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        output_filename = f"forecast_output_{now_str}.xlsx"
        output_dir = "files/output/"
        os.makedirs(output_dir, exist_ok=True)
        forecast_path = os.path.join(output_dir, output_filename)
        forecast_df.to_excel(forecast_path, index=False)
        print(f"✅ Итоговый прогноз сохранён в:", forecast_path)

        return forecast_path, model_name

    except Exception as e:
        print(f"❌ Ошибка при обработке и обучении модели: {e}")
        return None, None


def load_and_forecast(model_name: str):
    """
    Загружаем модель из БД

    В итоге 2 графика:
      - test_forecast_*
      - forecast_*

    """
    try:
        print(f"ℹ️ Пытаемся загрузить модель '{model_name}' из БД...")
        p = ProcessingData()

        # Ищем запись о модели
        with p.engine.begin() as conn:
            query = text("SELECT model_name, last_date FROM models WHERE model_name = :m")
            row = conn.execute(query, {"m": model_name}).fetchone()

        if not row:
            print(f"❌ Модель '{model_name}' не найдена в базе данных.")
            return None

        file_name, last_dt = row
        model_path = os.path.join("./models/", file_name)
        if not os.path.exists(model_path):
            print(f"❌ Файл модели '{model_path}' не найден.")
            return None

        print(f"ℹ️ Модель найдена. Загружаем из: {model_path}")
        data = p.get_weather_data()
        data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
        data = data.sort_values(by='date')
        series = data['temp'].values
        wf = WeatherForecastModel(p.engine, window_size=12)
        wf.load_model(model_path)

        print("ℹ️ Проверяем, достаточно ли данных для тест-графика (≥24)...")
        if len(series) >= 24:
            print("ℹ️ Делаем условный 'тест-прогноз' на последних 12 точках...")
            train_part = series[:-12]
            test_part = series[-12:]
            if len(train_part) >= wf.window_size:
                last_window_train = train_part[-wf.window_size:]
                test_preds = wf.forecast(last_window_train, horizon=12)
                # Будем сравнивать test_preds и test_part
                test_plot = wf.plot_test_forecast(test_part, test_preds, model_name + "_loaded")
                print("ℹ️ Построен тест-график:", test_plot)
            else:
                print("❗ Недостаточно данных для тест-прогноза (train_part < window_size). Пропускаем тест.")
        else:
            print("❗ Недостаточно данных (менее 24) — пропускаем тестовый прогноз.")

        print("ℹ️ Делаем финальный прогноз на 12 месяцев вперёд...")
        last_window = series[-wf.window_size:]
        future_df = wf.forecast_with_dates(last_window, last_dt, horizon=12)
        forecast_plot = wf.plot_forecast(future_df, model_name + "_loaded")
        print("ℹ️ Построен график прогноза:", forecast_plot)

        # Сохраним итоговый forecast в Excel (с датой)
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        output_filename = f"forecast_{model_name}_{now_str}.xlsx"
        output_dir = "files/output/"
        os.makedirs(output_dir, exist_ok=True)
        forecast_file = os.path.join(output_dir, output_filename)
        future_df.to_excel(forecast_file, index=False)
        print("✅ Прогноз сохранён в:", forecast_file)

        return forecast_file

    except Exception as e:
        print(f"❌ Ошибка при загрузке модели и прогнозировании: {e}")
        return None
