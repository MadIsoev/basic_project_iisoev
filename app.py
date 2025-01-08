import pickle

# Сохранение обученной модели
with open("car_price_model.pkl", "wb") as file:
    pickle.load(forest_model, file)
