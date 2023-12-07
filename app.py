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
                           'о клиенте и бронировании. Признаки: ' \
                           'lead_time - Время предварительного бронирования. ' \
                           'stays_in_weekend_nights - Количество проживаемых выходных ночей. ' \
                           'stays_in_week_nights - Количество проживаемых будних ночей.' \
                           'adults - Количество взрослых. ' \
                           'children - Количество детей. ' \
                           'babies - Количество младенцев. ' \
                           'meal - Тип обеда. ' \
                           'customer_type - Тип клиента. ' \
                           'previous_cancellations - Предыдущие отмены бронирования. ' \
                           'previous_bookings_not_canceled - Предыдущие успешные бронирования (не отмененные). ' \
                           'required_car_parking_spaces - Требуемое количество парковочных мест для автомобилей. ' \
                           'CPI_AVG - Средний индекс потребительских цен. ' \
                           'INFLATION - Инфляция. ' \
                           'INFLATION_CHG - Изменение инфляции. ' \
                           'GDP - ВВП (валовый внутренний продукт). ' \
                           'CPI_HOTELS - Индекс потребительских цен на отели. ' \
                           'Целевая переменная: ' \
                           'booking_canceled - Флаг отмены бронирования (1 - отменено, 0 - не отменено).'

description_for_2nd_task = 'Кластеризация - метод DBSCAN - кластеризация по lead_time (время до бронирования), ' \
                           'stays_in_weekend_nights (количество проживаемых выходных ночей), ' \
                           'adults (количество взрослых), children (количество детей), babies (количество младенцев), ' \
                           'adr (средняя цена за номер в день) - кластеризация клиентов по их характеристикам ' \
                           'Кластеризация - метод K-Means - кластеризация по lead_time (время до бронирования), ' \
                           'stays_in_weekend_nights (количество проживаемых выходных ночей), ' \
                           'adults (количество взрослых), children (количество детей), babies (количество младенцев), ' \
                           'adr (средняя цена за номер в день) - кластеризация клиентов по их характеристикам'

description_for_3rd_task = 'Гребневая регрессия - прогнозирование значения дохода (adr) на основе набора ' \
                           'экономических показателей: lead_time (время до бронирования), ' \
                           'stays_in_weekend_nights (количество проживаемых выходных ночей), ' \
                           'stays_in_week_nights (количество проживаемых будних ночей) ' \
                           'adults (количество взрослых), children (количество детей), babies (количество младенцев), ' \
                           'meal (тип обеда), customer_type (вид покупателя), ' \
                           'previous_cancellations (количество предыдущих отмен бронирования), ' \
                           'previous_bookings_not_canceled (количество предыдущих неотмененных бронирований), ' \
                           'required_car_parking_spaces (количество необходимых мест для парковки), ' \
                           'CPI_AVG (Средний индекс потребительских цен), INFLATION (Инфляция), ' \
                           'INFLATION_CHG (Изменение инфляции), GDP (валовый внутренний продукт, ВВП), ' \
                           'CPI_HOTELS (Индекс потребительских цен на отели) ' \
                           'Нейронная сеть - многослойный персептрон (MLPRegressor) - ' \
                           'прогнозирование значения дохода (adr) на основе различных ' \
                           'экономических и операционных показателей: lead_time (время до бронирования), ' \
                           'stays_in_weekend_nights (количество проживаемых выходных ночей), ' \
                           'stays_in_week_nights (количество проживаемых будних ночей) ' \
                           'adults (количество взрослых), children (количество детей), babies (количество младенцев), ' \
                           'meal (тип обеда), customer_type (вид покупателя), ' \
                           'previous_cancellations (количество предыдущих отмен бронирования), ' \
                           'previous_bookings_not_canceled (количество предыдущих неотмененных бронирований), ' \
                           'required_car_parking_spaces (количество необходимых мест для парковки), ' \
                           'CPI_AVG (Средний индекс потребительских цен), INFLATION (Инфляция), ' \
                           'INFLATION_CHG (Изменение инфляции), GDP (валовый внутренний продукт, ВВП), ' \
                           'CPI_HOTELS (Индекс потребительских цен на отели)'

list_plot_urls = ['images/decision_tree.png', 'images/decision_tree_graph.png',
                  'images/confusion_matrix_tree_task1.png', 'images/confusion_matrix_task1.png',
                  'images/neural_task1.png', 'images/clusters_dbscan.png',
                  'images/elbow_method.png', 'images/clusters_kmeans.png',
                  'images/neural_task3.png', 'images/ridge.png']

list_descriptions = {
    'hotel': 'показывает название отеля, в котором производилось изучение данных',
    'is_canceled': 'флаг отмены бронирования (1 - отменено, 0 - не отменено)',
    'lead_time': 'время предварительного бронирования',
    'arrival_date_year': 'год прибытия',
    'arrival_date_month': 'месяц прибытия',
    'arrival_date_week_number': 'номер недели прибытия',
    'arrival_date_day_of_month': 'день месяца прибытия',
    'stays_in_weekend_nights': 'количество проживаемых выходных ночей',
    'stays_in_week_nights': 'количество проживаемых будних ночей',
    'adults': 'количество взрослых',
    'children': 'количество детей',
    'babies': 'количество младенцев',
    'meal': 'тип обеда',
    'country': 'страна происхождения клиента',
    'market_segment': 'сегмент рынка, к которому относится клиент',
    'distribution_channel': 'канал распределения',
    'is_repeated_guest': 'флаг повторного посещения (1 - повторный гость, 0 - новый гость)',
    'previous_cancellations': 'предыдущие отмены бронирования клиента',
    'previous_bookings_not_canceled': 'предыдущие успешные бронирования (не отмененные)',
    'reserved_room_type': 'тип зарезервированного номера',
    'assigned_room_type': 'тип назначенного номера',
    'booking_changes': 'количество изменений в бронировании',
    'deposit_type': 'тип депозита, оплаченного клиентом',
    'agent': 'идентификатор агента, совершившего бронирование',
    'days_in_waiting_list': 'количество дней в списке ожидания',
    'customer_type': 'тип клиента',
    'adr': 'средняя цена за номер в день',
    'required_car_parking_spaces': 'требуемое количество парковочных мест для автомобилей',
    'total_of_special_requests': 'общее количество специальных запросов клиента',
    'reservation_status': 'статус бронирования',
    'reservation_status_date': 'дата обновления статуса бронирования',
    'MO_YR': 'месяц и год прибытия',
    'CPI_AVG': 'средний индекс потребительских цен',
    'INFLATION': 'инфляция',
    'INFLATION_CHG': 'изменение инфляции',
    'CSMR_SENT': 'потребительские настроения',
    'UNRATE': 'уровень безработицы',
    'INTRSRT': 'процентная ставка',
    'GDP': 'валовый внутренний продукт',
    'FUEL_PRCS': 'цена на топливо',
    'CPI_HOTELS': 'индекс потребительских цен на отели',
    'US_GINI': 'индекс Джини для США',
    'DIS_INC': 'распределение доходов'
}


@app.route('/')
def home():
    # Первые 10 строк для представления
    first_10_rows = df.head(10).copy()
    column_names = df.columns.tolist()

    return render_template('index.html', rows=first_10_rows, columns=column_names,
                           variable_descriptions=list_descriptions)


@app.route('/task1')
def decision_tree():
    result_tree = decision_tree_task1(df)
    result_neural = neural_network_task1(df)
    return render_template('task1.html', description=description_for_1st_task, plot_urls=list_plot_urls,
                           tree_quality=result_tree['accuracy'], features_importance=result_tree['features'],
                           neural_quality=result_neural)


@app.route('/task2')
def clustering():
    clustering_dbscan_task2(df)
    clustering_kmeans_task2(df)
    return render_template('task2.html', description=description_for_2nd_task, plot_urls=list_plot_urls)


@app.route('/task3')
def neural_network():
    neural_network_task3(df)
    ridge_regression_task3(df)
    return render_template('task3.html', description=description_for_3rd_task, plot_urls=list_plot_urls)


if __name__ == '__main__':
    app.run(debug=True)
