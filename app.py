import streamlit as st
import pickle
import numpy as np

# Загрузка модели
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Функция для предсказания цены
def predict_price(model, data):
    return model.predict(data)

# Интерфейс Streamlit
st.title("Предсказание цены автомобиля")
st.write("Введите параметры автомобиля, чтобы получить предсказанную цену (в сомони).")

# Ввод данных
model_name = st.text_input("Модель автомобиля", "Toyota")
year = st.slider("Год выпуска", 2000, 2025, 2015)
transmission = st.selectbox("Коробка передач", ['Автомат', 'Механика', 'Робот', 'Вариатор'])
fuel_type = st.selectbox("Вид топлива", ['Дизель', 'Бензин', 'Бензин + газ', 'Газ'])
city = st.text_input("Город", "Душанбе")

# Преобразование данных для модели
# Пример преобразования данных (должно соответствовать обучению модели)
input_data = np.array([[year, transmission, fuel_type]])  # Убедитесь, что city не используется моделью, если это так.

# Кнопка для предсказания
if st.button("Предсказать цену"):
    try:
        # Предсказание
        price = predict_price(model, input_data)
        st.success(f"Предсказанная цена автомобиля: {price[0]:,.2f} сомони")
    except Exception as e:
        st.error(f"Ошибка: {e}")
