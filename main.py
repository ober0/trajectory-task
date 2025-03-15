import pandas as pd

from processingData import ProcessingData
from forecastModel import WeatherForecastModel


if __name__ == '__main__':

    processingData = ProcessingData()
    processingData.parseData('weather-data.xlsx')

    weather_data = processingData.get_weather_data()

    # Преобразуем данные в одномерный массив температур для обучения
    weather_data['date'] = pd.to_datetime(weather_data[['year', 'month']].assign(day=1))
    weather_data = weather_data.sort_values(by='date')

    # Извлекаем только температурные данные
    series = weather_data['temp'].values

    # Инициализируем модель для прогнозирования
    wf_model = WeatherForecastModel(processingData.engine, window_size=12)

    # Обучаем модель
    wf_model.train_model(series, epochs=50, batch_size=8)

    # Сохраняем модель и записываем её имя в БД
    wf_model.save_model('weather_model.h5')

    # Формируем окно для прогнозирования – последние 12 месяцев
    last_window = series[-wf_model.window_size:]

    # Определяем последнюю дату наблюдения (например, декабрь 2024 года)
    last_date = '2025-03-01'

    # Получаем прогноз на следующие 12 месяцев
    forecast_df = wf_model.forecast_with_dates(last_window, last_date, forecast_horizon=12)

    # Выводим прогноз
    print(forecast_df)