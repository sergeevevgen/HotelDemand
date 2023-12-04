# Импортируем необходимые библиотеки
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Загрузим данные
df = pd.read_csv("hotel_bookings_raw.csv")
df.dropna(inplace=True)

# Объявляю объект для преобразования строковых значений в числовые
label_encoder = LabelEncoder()

# Выберем признаки и целевую переменную (доход)
features_list = ['lead_time', 'stays_in_weekend_nights',
                 'stays_in_week_nights', 'adults', 'children', 'babies', 'meal', 'customer_type',
                 'previous_cancellations',
                 'previous_bookings_not_canceled', 'required_car_parking_spaces',
                 'CPI_AVG', 'INFLATION', 'INFLATION_CHG', 'GDP', 'CPI_HOTELS']
features = df[features_list].copy()

# Применяю к каждому столбцу признака преобразования
for f in features_list:
    features[f] = label_encoder.fit_transform(features[f])

target = df['adr'].copy()

# Разделим данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Масштабируем признаки для лучшей производительности модели
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создаем модель гребневой регрессии
ridge_model = Ridge(alpha=1.0)  # Можете изменить alpha в зависимости от необходимой регуляризации

# Обучаем модель на тренировочных данных
ridge_model.fit(X_train_scaled, y_train)

# Делаем предсказания на тестовых данных
predictions = ridge_model.predict(X_test_scaled)

# Оцениваем производительность модели по MSE метрике
mse = math.sqrt(mean_squared_error(y_test, predictions))

# Оцениваем производительность модели по MAE метрике
mae = mean_absolute_error(y_test, predictions)

# Оцениваем производительность модели по R^2 метрике
r2 = r2_score(y_test, predictions)

print(f"Среднеквадратичная ошибка (MSE): {round(mse, 2)}%")
print(f"Среднеабсолютное отклонение (MAE): {round(mae, 2)}%")
print(f"Коэффициент детерминации (R^2): {round(r2, 4) * 100}%")