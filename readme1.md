# Лабораторная работа 1. Работа с типовыми наборами данных и различными моделями
## Задание
Сгенерировать определенный тип данных и сравнить на нем 3 модели. Построить графики, отобразить качество моделей,
объяснить полученные результаты.
Вариант 3 (24)
Данные: make_classification 
(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=rs, n_clusters_per_class=1) 
Модели:
· Линейную регрессию
· Полиномиальную регрессию (со степенью 3)
· Гребневую полиномиальную регрессию (со степенью 3, alpha= 1.0)


### Запуск программы
Файл lab1.py содержит и запускает программу

### Описание программы
Генерирует набор данных, показывает окно с графиками и пишет среднюю ошибку моделей обучения
Использует библиотеки: matplotlib для демонстрации графиков и sklearn для создания и использования моделей. 

### Результаты тестирования
Для значения rs=10 результаты такие:
y - linear_y  - polyn_y - ridge_y
0 -   0.092   -  0.058  -  0.062
0 -   0.023   -  -0.132 -  -0.125
1 -   1.32    -  0.789  -  0.8
1 -   0.84    -  1.068  -  1.06

### Вывод
Из представленных данных можно сделать вывод,
что линейная регрессия и гребневая регрессия,
в целом, предсказывают значения, близкие к исходным,
и хорошо справляются с задачей. Полиномиальная регрессия 
иногда может давать менее точные прогнозы, особенно когда данные имеют сложную структуру. 