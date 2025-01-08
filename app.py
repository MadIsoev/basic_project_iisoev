{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5e1fce0b-627a-471f-a816-5f0f27cad3e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Загрузка модели\n",
    "with open(\"car_price_model.pkl\", \"rb\") as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "# Функция для предсказания цены\n",
    "def predict_price(model, data):\n",
    "    return model.predict(data)\n",
    "\n",
    "# Интерфейс Streamlit\n",
    "st.title(\"Предсказание цены автомобиля\")\n",
    "st.write(\"Введите параметры автомобиля, чтобы получить предсказанную цену (в сомони).\")\n",
    "\n",
    "# Ввод данных\n",
    "model_name = st.text_input(\"Модель автомобиля\", \"Toyota\")\n",
    "year = st.slider(\"Год выпуска\", 2000, 2025, 2015)\n",
    "transmission = st.selectbox(\"Коробка передач\", ['Автомат', 'Механика', 'Робот', 'Вариатор'])\n",
    "fuel_type = st.selectbox(\"Вид топлива\", ['Дизель', 'Бензин', 'Бензин + газ', 'Газ'])\n",
    "city = st.text_input(\"Город\", \"Душанбе\")\n",
    "\n",
    "# Преобразование данных для модели\n",
    "# Если использовался LabelEncoder, преобразуем строковые значения обратно в числа\n",
    "input_data = np.array([[model_name, year, transmission, fuel_type, city]])\n",
    "\n",
    "# Важно: Убедитесь, что входные данные масштабированы так же, как при обучении\n",
    "# Например, если использовалась нормализация данных, сделайте это:\n",
    "# input_data_scaled = scaler.transform(input_data) если был использован scaler.\n",
    "\n",
    "# Кнопка для предсказания\n",
    "if st.button(\"Предсказать цену\"):\n",
    "    # Предсказание\n",
    "    price = predict_price(model, input_data)\n",
    "    st.success(f\"Предсказанная цена автомобиля: {price[0]:,.2f} сомони\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "85bb18ea-a236-4a39-bf71-ae2a16388df8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
