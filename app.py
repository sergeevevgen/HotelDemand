import pandas as pd
from flask import Flask, render_template
from firstExerciseWithDecision_tree import decision_tree_task1
from firstExerciseWithNeuralNetwork import neural_network_task1
from secondExerciseWithDBSCAN import clustering_dbscan_task2
from secondExerciseWithKMeans import clustering_kmeans_task2
from thirdExerciseWithNeuralNetwork import neural_network_task3
from thirdExerciseWithRidgeRegression import ridge_regression_task3

app = Flask(__name__)

df = pd.read_csv("hotel_bookings_raw.csv", delimiter=',')
df.dropna(inplace=True)

description_for_1st_task = 'Дерево решений - классификация - определение вероятности отмены бронирования на основе ' \
                           'данных ' \
                           'о клиенте и бронировании. Признаки:' \
                           ' lead_time - Время предварительного бронирования. ' \
                           'stays_in_weekend_nights - Количество проживаемых выходных ночей.' \
                           'stays_in_week_nights - Количество проживаемых будних ночей.' \
                           'adults - Количество взрослых.' \
                           'children - Количество детей.' \
                           'babies - Количество младенцев.' \
                           'meal - Тип обеда.' \
                           'customer_type - Тип клиента.' \
                           'previous_cancellations - Предыдущие отмены бронирования.' \
                           'previous_bookings_not_canceled - Предыдущие успешные бронирования (не отмененные).' \
                           'required_car_parking_spaces - Требуемое количество парковочных мест для автомобилей.' \
                           'CPI_AVG - Средний индекс потребительских цен.' \
                           'INFLATION - Инфляция.' \
                           'INFLATION_CHG - Изменение инфляции.' \
                           'GDP - ВВП (валовый внутренний продукт).' \
                           'CPI_HOTELS - Индекс потребительских цен на отели.' \
                           'Целевая переменная:' \
                           'booking_canceled - Флаг отмены бронирования (1 - отменено, 0 - не отменено).'

list_plot_urls = ['static/images/decision_tree.png', 'static/images/imagesconfusion_matrix_task1.png',
                  'static/images/neural_task1.png', 'static/images/clusters_dbscan.png',
                  'static/images/elbow_method.png', 'static/images/clusters_kmeans.png',
                  'static/images/neural_task3.png', 'static/images/ridge.png']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/task1')
def decision_tree():
    decision_tree_task1(df)
    return render_template('task1.html', description=description_for_1st_task, plot_urls=list_plot_urls)


@app.route('/clustering')
def clustering():
    # Здесь можно вставить код для выполнения задачи с кластеризацией
    return render_template('task1.html')


@app.route('/task3')
def neural_network():
    # Здесь можно вставить код для выполнения задачи с нейронной сетью
    return render_template('task1.html')


if __name__ == '__main__':
    app.run(debug=True)
