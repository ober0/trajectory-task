
from main import process_and_train_model, load_and_forecast

def console_loop():
    while True:
        print("\n--- МЕНЮ ---")
        print(" 1 - Обучить новую модель")
        print(" 2 - Загрузить существующую модель и сделать прогноз")
        print(" q - Выйти из приложения\n")

        choice = input("Введите команду: ").strip().lower()

        if choice == '1':
            # Обучить новую модель
            try:
                input_file = input("Введите имя файла с данными (например, 'weather-data.xlsx'): ").strip()
                if not input_file:
                    print("❌ Ошибка: не указано имя файла.")
                    continue

                model_base_name = input("Введите базовое имя модели (например, 'test-model'): ").strip()
                if not model_base_name:
                    print("❌ Ошибка: не указано название модели.")
                    continue

                forecast_path, model_name = process_and_train_model(input_file, model_base_name)
                if forecast_path and model_name:
                    print("✅ Модель обучена! Прогноз сохранён в:", forecast_path)
                    print("✅ Имя модели (для использования в будущем):", model_name)
                else:
                    print("❌ Произошла ошибка при обучении или сохранении модели.")

            except KeyboardInterrupt:
                print("\nОперация прервана пользователем.")
            except Exception as e:
                print(f"❌ Непредвиденная ошибка при обучении: {e}")

        elif choice == '2':
            # Загрузить модель и сделать прогноз на 12 месяцев
            try:
                model_name = input("Введите точное имя сохранённой модели (например, 'test-model_1679999999.keras'): ").strip()
                if not model_name:
                    print("❌ Ошибка: не указано имя модели.")
                    continue

                forecast_file = load_and_forecast(model_name)
                if forecast_file:
                    print("✅ Прогноз успешно сформирован и сохранён в:", forecast_file)
                else:
                    print("❌ Ошибка при загрузке модели или генерации прогноза.")

            except KeyboardInterrupt:
                print("\nОперация прервана пользователем.")
            except Exception as e:
                print(f"❌ Непредвиденная ошибка при прогнозировании: {e}")

        elif choice == 'q':
            # Выходим из бесконечного цикла
            print("Завершение работы...")
            break

        else:
            print("❌ Неизвестная команда. Повторите ввод или нажмите 'q' для выхода.")

def main():
    """
    Точка входа в консольное приложение.
    """
    print("Добро пожаловать в консольное приложение прогноза погоды!\n")
    try:
        console_loop()
    except KeyboardInterrupt:
        print("\nПриложение остановлено пользователем.")
    except Exception as e:
        print(f"❌ Непредвиденная ошибка в приложении: {e}")
    finally:
        print("Выход из приложения.")

if __name__ == '__main__':
    main()
