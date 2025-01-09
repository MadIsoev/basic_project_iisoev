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


