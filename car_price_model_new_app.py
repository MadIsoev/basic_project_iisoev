import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

st.title('Предсказание цены на подержанные автомобили Opel')

st.write('Прогнозирование цены основывается на таких признаках, как модель, год выпуска, коробка передач, тип топлива и город, где находится автомобиль.')

# Загрузка данных
df = pd.read_csv("Opel_data.csv")

# Удаляем разделители тысяч и преобразуем в числовые данные
df['Year'] = df['Year'].astype(int)
df['Price'] = df['Price'].astype(int)

# Отображение данных
with st.expander('Данные'):
    st.write("Признаки (X):")
    X_raw = df.drop('Price', axis=1)
    st.dataframe(X_raw)

    st.write("Целевая переменная (y):")
    y_raw = df['Price']
    st.dataframe(y_raw)

# Ввод данных пользователем
with st.sidebar:
    st.header("Введите характеристики автомобиля: ")
    model = st.selectbox('Модель', df['Model'].unique())
    year = st.slider('Год выпуска', int(df['Year'].min()), int(df['Year'].max()))
    transmission = st.selectbox('Коробка передач', df['Transmission'].unique())
    fuel_type = st.selectbox('Тип топлива', df['Fuel type'].unique())
    city = st.selectbox('Город', df['City'].unique())

# Визуализация данных
st.subheader('Визуализация данных')
fig = px.scatter(df, x='Year', y='Price', color='Fuel type', title='Зависимость цены от года выпуска и типа топлива')
st.plotly_chart(fig)

fig2 = px.histogram(df, x='Price', nbins=30, title='Распределение цен автомобилей')
st.plotly_chart(fig2)

# Предобработка данных
data = {
    'Model': model,
    'Year': year,
    'Transmission': transmission,
    'Fuel type': fuel_type,
    'City': city
}
input_df = pd.DataFrame([data])
combined_df = pd.concat([input_df, X_raw], axis=0)

# Отображение введённых данных
with st.expander('Введённые данные'):
    st.write('Введённые характеристики автомобиля:')
    st.dataframe(input_df)

# Кодирование категориальных признаков
encoded_df = pd.get_dummies(combined_df, columns=['Model', 'Transmission', 'Fuel type', 'City'], drop_first=True)

# Разделение данных
X = encoded_df.iloc[1:].reset_index(drop=True)
input_row = encoded_df.iloc[0:1]

# Целевая переменная
y = y_raw

# Подготовка данных
with st.expander('Подготовленные данные'):
    st.write('**Признаки (X):**')
    st.dataframe(X)
    st.write('**Целевая переменная (y):**')
    st.write(y)
