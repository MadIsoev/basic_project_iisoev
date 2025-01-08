import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Заголовок приложения
st.title('Предсказание цены на подержанные автомобили Opel')

st.write('Прогнозирование цены основывается на таких признаках, как модель, год выпуска, коробка передач, тип топлива и город, где находится автомобиль.')

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("Opel_data.csv")

# Обучение модели
@st.cache_resource
def train_model(df):
    label_encoders = {}
    for col in ['Model', 'Transmission', 'Fuel type', 'City']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Разделение данных
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Обучение модели
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, 'opel_model.pkl')
    return model, label_encoders

# Загрузка данных
df = load_data()

# Преобразование числовых данных
df['Year'] = df['Year'].astype(int)
df['Price'] = df['Price'].astype(int)

# Визуализация данных
st.subheader('Визуализация данных')
fig = px.scatter(df, x='Year', y='Price', color='Fuel type', title='Зависимость цены от года выпуска и типа топлива')
st.plotly_chart(fig)

fig2 = px.histogram(df, x='Price', nbins=30, title='Распределение цен автомобилей')
st.plotly_chart(fig2)

# Обучение модели
model, label_encoders = train_model(df)

# Интерфейс для ввода данных
with st.sidebar:
    st.header("Введите характеристики автомобиля: ")
    model_input = st.selectbox('Модель', label_encoders['Model'].classes_)
    year_input = st.slider('Год выпуска', min(df['Year']), max(df['Year']), step=1)
    transmission_input = st.selectbox('Коробка передач', label_encoders['Transmission'].classes_)
    fuel_type_input = st.selectbox('Тип топлива', label_encoders['Fuel type'].classes_)
    city_input = st.selectbox('Город', label_encoders['City'].classes_)

# Преобразование введённых данных
model_encoded = label_encoders['Model'].transform([model_input])[0]
transmission_encoded = label_encoders['Transmission'].transform([transmission_input])[0]
fuel_type_encoded = label_encoders['Fuel type'].transform([fuel_type_input])[0]
city_encoded = label_encoders['City'].transform([city_input])[0]

input_data = pd.DataFrame(
    [[model_encoded, year_input, transmission_encoded, fuel_type_encoded, city_encoded]],
    columns=['Model', 'Year', 'Transmission', 'Fuel type', 'City']
)

# Отображаем в Streamlit
with st.expander('Введённые данные'):
    st.dataframe(pd.DataFrame([decoded_data]))


# Предсказание цены
if st.button("Предсказать цену"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Прогнозируемая цена автомобиля: {prediction:.2f} сомони")
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")

# Дополнительно: вывод вероятностей или других показателей
with st.expander('Детали модели и вероятности'):
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=input_data.columns,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)
    st.write("**Важность признаков:**")
    st.dataframe(feature_importances)
