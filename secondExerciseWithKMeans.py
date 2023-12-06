import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# Загружаю данные
# df = pd.read_csv("hotel_bookings_raw.csv", delimiter=',')
# df.dropna(inplace=True)

def clustering_kmeans_task2(df):
    # Объявляю объект для преобразования строковых значений в числовые
    label_encoder = LabelEncoder()

    # Признаки, по которым будет проходить кластеризация
    features = ['lead_time', 'stays_in_weekend_nights', 'adults', 'children', 'babies', 'adr']

    df = df[features].copy()
    # Применяю к каждому столбцу признака преобразования
    for f in features:
        df[f] = label_encoder.fit_transform(df[f])

    # Стандартизация данных
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Определение числа кластеров с помощью метода локтя (Elbow Method)
    # inertia = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, random_state=42)
    #     kmeans.fit(scaled_features)
    #     inertia.append(kmeans.inertia_)
    #
    # # Визуализация метода локтя
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 11), inertia, marker='o')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Inertia')

    # Сохранение графика в файл .png
    # plt.savefig('clustersKMeans.png', dpi=300)
    # plt.show()

    # Исходя из графика, определяем оптимальное количество кластеров - определяем сами
    n_clusters = 5  # Выбираем количество кластеров - 4 или 5 - самое оптимальное

    # Кластеризация с использованием K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # Визуализация результатов кластеризации
    # sns.boxplot(x='cluster', y='adr', data=df, palette='Dark2')
    sns.violinplot(x='cluster', y='adr', data=df, palette='Dark2')
    plt.savefig('clustersKMeans.png', dpi=300)
    plt.show()
