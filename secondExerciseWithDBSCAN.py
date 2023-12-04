import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Загружаю данные
df = pd.read_csv("hotel_bookings_raw.csv", delimiter=',')

# Объявляю объект для преобразования строковых значений в числовые
label_encoder = LabelEncoder()

# Признаки, по которым будет проходить кластеризация
features = ['lead_time', 'stays_in_weekend_nights', 'adults', 'children', 'babies', 'adr']

# Применяю к каждому столбцу признака преобразования
for f in features:
    df[f] = label_encoder.fit_transform(df[f])

# Создаю объект для стандартизации данных
scaler = StandardScaler()

# Стандартизую признаки
scaled_features = scaler.fit_transform(df[features])

# Создаю объект для метода кластеризации DBSCAN
# Это алгоритм кластеризации,
# основанной на плотности —
# если дан набор точек в
# некотором пространстве, алгоритм
# группирует вместе точки, которые тесно расположены
# (точки со многими близкими соседями[en]), помечая как выбросы
# точки, которые находятся одиноко в областях с малой плотностью
# (ближайшие соседи которых лежат далеко). DBSCAN является одним
# из наиболее часто используемых алгоритмов кластеризации, и наиболее часто упоминается в научной литературе
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Кластеризую данные по этим признакам
clusters = dbscan.fit_predict(scaled_features)

# Создаю график
plt.scatter(df[features[0]], df[features[1]], c=clusters)
plt.title('Метод кластеризации - DBSCAN')
plt.xlabel('Время до заезда')
plt.ylabel('Забронировано ночей в выходные дни')

# Сохранение графика в файл .png
plt.savefig('clustersDBSCAN.png', dpi=300)
