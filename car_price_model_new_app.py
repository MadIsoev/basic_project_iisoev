#Первая часть
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

st.title('Предсказание цены на подержанные автомобили Opel')

st.write('Прогнозирование цены основывается на таких признаках, как модель, год выпуска, коробка передачи, тип топлива и город, где находится автомобиль.')

# Пример загрузки данных
df = pd.read_csv("Opel_data.csv")

# Удаляем разделители тысяч в Year и Price
df['Year'] = df['Year'].apply(lambda x: f'{int(x)}')
df['Price'] = df['Price'].apply(lambda x: f'{int(x)}')

# Отображаем таблицу
with st.expander('Data'):
    st.write("Признаки:")
    X_raw = df.drop('Price', axis=1)
    st.dataframe(X_raw)

    st.write("Целовая переменная")
    y_raw = df['Price']
    st.dataframe(y_raw)

with st.sidebar:
  st.header("Введите характеристики автомобиля: ")
  model = st.selectbox('Модель', ('Opel Combo', 'Opel Astra H', 'Opel Astra G', 'Opel Astra F', 'Opel Vectra A', 'Opel Vectra B', 'Opel Vectra C', 'Opel Zafira', 
                                  'Opel Astra J', 'Opel Meriva', 'Opel Omega', 'Opel Frontera', 'Opel Astra K', 'Opel Insignia', 'Opel Vita', 'Opel Corsa', 'Opel Calibra', 
                                  'Opel Signum', 'Opel Tigra', 'Opel Antara', 'Opel Sintra', 'Opel Vectra С', 'Opel Vectra А', 'Opel Agila', 'Opel Mokka', 'Opel Campo', 
                                  'Opel Cavalier'))
  year = st.slider('Год выпуска', 1956, 2024, format="%d")
  transmission = st.selectbox('Коробка передачи', ('Автомат', 'Механика', 'Робот', 'Вариатор'))
  fuel_type = st.selectbox('Тип топлива', ('Дизель', 'Бензин', 'Бензин + газ', 'Газ', 'Другой'))
  city = st.selectbox('Город', ('Душанбе', 'Худжанд', 'Куляб', 'Хорог', 'Дангара', 'Яван', 'Пенджикент', 'Истаравшан', 'Кабодиён', 'Фархор', 'Вахдат', 'Рашт',
       'Дусти (Джиликуль)', 'Бободжон Гафуров', 'Файзабад', 'Ашт', 'Спитамен', 'Вахш', 'Исфара', 'Хамадани', 'Бохтар (Курган-Тюбе)',
       'Кушониён (Бохтар)', 'Рудаки', 'Пяндж', 'Канибадам', 'Хуросон', 'Шахринав', 'Джалолиддин Балхи (Руми)', 'Восе', 'Нурек',
       'Турсунзаде', 'Матча', 'Джаббор Расулов', 'Зафарабад', 'Джайхун (Кумсангир)', 'Деваштич (Ганчи)', 'Шахристон', 'Гиссар',
       'Варзоб', 'Гулистон (Кайраккум)', 'Абдурахмони Джоми', 'Шахритус', 'Бустон (Чкаловск)', 'Темурмалик', 'Леваканд (Сарбанд)',
       'Таджикабад', 'Рогун', 'Нурабад', 'Муминабад', 'Айни', 'Носири Хусрав', 'Джами', 'Лахш (Джиргиталь)',
       'Шамсиддин Шохин (Шуроабад)', 'Вандж', 'Ховалинг', 'Бальджувон', 'Горная Матча', 'Истиклол', 'Дарваз'))

# Визуализация данных
st.subheader('Визуализация данных')
fig = px.scatter(df, x='Year', y='Price', color='Fuel type', title='Зависимость цены от года выпуска и типа топлива')
st.plotly_chart(fig)

fig2 = px.histogram(df, x='Price', nbins=30, title='Распределение цен автомобилей')
st.plotly_chart(fig2)

## Preprocessing
data = {
    'Model': model,
    'Year': year,
    'Transmission': transmission,
    'Fuel type': fuel_type,
    'City': city
}

input_df = pd.DataFrame(data, index=[0])
input_cars = pd.concat([input_df, X_raw], axis=0)

# Вторая часть

with st.expander('Input features'):
    st.write('**Input car**')
    st.dataframe(input_df)
    st.write('**Combined car data** (input row + original data)')
    st.dataframe(input_cars)

# Кодирование категориальных признаков
encode = ['Model', 'Transmission', 'Fuel type', 'City']
input_cars_encoded = pd.get_dummies(input_cars, columns=encode)

# Разделяем входной автомобиль и остальные данные
X = input_cars_encoded[1:]
input_row = input_cars_encoded[:1]

# Целевая переменная (цена)
y = y_raw

with st.expander('Data preparation'):
    st.write('**Encoded X (input car)**')
    st.dataframe(input_row)
    st.write('**Encoded y**')
    st.write(y)

# Обучение модели
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

# Создаем базовую модель
base_rf = RandomForestClassifier(random_state=42)

# Проводим Grid Search
grid_search = GridSearchCV(base_rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
st.write("**Best Parameters**:", best_params)

# Применяем модель для предсказания
prediction = best_model.predict(input_row)

# Вывод результата
st.subheader('Predicted Price')
st.success(f"Predicted price for the selected car: **{prediction[0]}** сомони")

