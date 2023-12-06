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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/decision_tree')
def decision_tree():
    # Здесь можно вставить код для выполнения задачи с деревом решений
    return "Решение задачи с использованием дерева решений"


@app.route('/clustering')
def clustering():
    # Здесь можно вставить код для выполнения задачи с кластеризацией
    return "Решение задачи с использованием кластеризации"


@app.route('/neural_network')
def neural_network():
    # Здесь можно вставить код для выполнения задачи с нейронной сетью
    return "Решение задачи с использованием нейронной сети"


if __name__ == '__main__':
    app.run(debug=True)
