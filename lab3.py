import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

df = pd.read_csv("hotel_bookings_raw.csv", delimiter=',')


# Тип питания в зависимости от инфляции и потребительских настроений по отношению к экономике
# Функция для отображения дерева решений
def tree_visual():
    # Преобразование типов питания к числам
    # Инициализация LabelEncoder
    label_encoder = LabelEncoder()

    # Преобразование столбцов в числовые значения для всего фрейма
    # list_col = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type', 'reservation_status', 'assigned_room_type']
    # for i in list_col:
    #     df[i] = label_encoder.fit_transform(df[i])
    #df[list_col] = label_encoder.fit_transform(df[list_col])

    #для всего фрейма
    # mean_value = round(df['agent'].mean())
    # df['agent'].fillna(mean_value, inplace=True)

    # Количество элементов для обучения (99%)
    count_to_train = round(len(df) * 0.99)
    # Количество элементов для тестирования (1%)
    count_to_test = len(df) - count_to_train

    # Набор данных для обучения для всего фрейма
    # train_df = df.head(count_to_train).copy().drop(columns=['reservation_status_date', 'MO_YR'])
    train_df = df.head(count_to_train).copy()

    # Тип питания
    y = train_df.copy()['meal']

    # Уровень инфляции и потребительские настроения по отношению к экономике для всего фрейма
    # x = train_df.copy().drop(columns=['meal'])
    x = train_df.copy()[['INFLATION', 'CSMR_SENT']]
    # Создание модели дерева решений
    model = DecisionTreeClassifier()

    # Обучение модели
    model.fit(x, y)

    # Проверка модели для всего фрейма
    #test_df = df.tail(count_to_test).copy().drop(columns=['reservation_status_date', 'MO_YR'])
    test_df = df.tail(count_to_test)[['INFLATION', 'CSMR_SENT', 'meal']]

    y_test = test_df.copy()['meal']
    x_test = test_df.copy().drop(columns=['meal'])

    prediction = model.score(x_test, y_test)

    print('Качество дерева решений: ', prediction * 100, '%')
    # Визуализация дерева решений
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=['INFLATION', 'CSMR_SENT'], filled=True)

    # Сохранение графика в файл .png
    plt.savefig('decision_tree.png', dpi=300)
    plt.show()

    res = sorted(dict(zip(list(x.columns), model.feature_importances_)).items(),
                 key=lambda el: el[1], reverse=True)

    flag = 0
    print('feature importance:')
    for val in res:
        print(val[0]+" - "+str(val[1]))

    return


tree_visual()
