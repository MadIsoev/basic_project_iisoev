import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Загрузка модели
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Инициализация LabelEncoder для категориальных данных
transmission_encoder = LabelEncoder()
fuel_type_encoder = LabelEncoder()
city_encoder = LabelEncoder()
model_name_encoder = LabelEncoder()

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

# Преобразование категориальных данных в числовые
model_name_encoded = model_name_encoder.fit_transform([model_name])[0]
transmission_encoded = transmission_encoder.fit_transform([transmission])[0]
fuel_type_encoded = fuel_type_encoder.fit_transform([fuel_type])[0]
city_encoded = city_encoder.fit_transform([city])[0]

# Преобразование данных для модели
input_data = np.array([[model_name_encoded, year, transmission_encoded, fuel_type_encoded, city_encoded]])

# Важно: Убедитесь, что входные данные масштабированы так же, как при обучении
# Пример: input_data_scaled = scaler.transform(input_data) если был использован scaler

# Кнопка для предсказания
if st.button("Предсказать цену"):
    # Предсказание
    price = predict_price(model, input_data)
    st.success(f"Предсказанная цена автомобиля: {price[0]:,.2f} сомони")
