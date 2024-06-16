# Проект "Ледокол": Оптимизация расписания проводки судов во льдах

## Содержание
1. [Описание проекта](#description)
2. [Описание данных](#data)
3. [Описание этапов работы](#stages)
4. [Результаты работы](#results)
5. [Используемые библиотеки](#libs)

## Задача и сложности проекта

## Архитектура бакенда проекта



## Алгоритм составления графа проходимости на основе данных о льдах

0. Образуем данные проходимости льда и разбиваем их по сетке координат.
1. Находим две точки на графе и берем их координаты. 
2. Строим путь между этими точками по болшому кругу <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Illustration_of_great-circle_distance.svg/220px-Illustration_of_great-circle_distance.svg.png" width="400">
3. Разбиваем путь на n точек
4. Для каждой точки на пути находим ближайшую точку данных по ледовой обстановке с помощью алгоритма K-d деревьев.
   1. Картинка
5. Строим граф, добавляя на рёбра метаданные о проходимости на участках пути.
6. Сохраняем.

## Алгоритм поиска оптимального пути на графе

The idea of Dijkstra’s algorithm is really easy. Suppose we drop a huge colony of ants onto the source node uuu at time 000. They split off from there and follow all possible paths through the graph at a rate of one unit per second. Then the first ant who finds the target node vvv will do so at time d(u,v)d(u,v)d(u, v) seconds, where d(u,v)d(u,v)d(u, v) is the shortest distance from uuu to vvv. How do we find when that is? We just need to watch the expanding wavefront of ants.

<img src="https://qph.cf2.quoracdn.net/main-qimg-cfa6e0006734b0bd93431c754a8c42c4" width="400">


### Симуляция отжига для оптимизации расписания
<img src="https://upload.wikimedia.org/wikipedia/commons/1/10/Travelling_salesman_problem_solved_with_simulated_annealing.gif" width="400">


## Симуляция отжига для оптимизации расписания

## Интеграция бакенда

## Описание API и примеры использования

## На что не хватило времени

### Создание навигационного решения основываясь на более подробных спутниковых снимков, данных о погоде и течениях

### Проводка судов без ледокола

### Создание картежей
