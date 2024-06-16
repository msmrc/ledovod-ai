# main.py

from fastapi import FastAPI, UploadFile, File
from .utils import *
import pandas as pd
import numpy as np

app = FastAPI()

@app.post("/update_graph")
async def update_graph(file: UploadFile = File(...)):
    # Сохраняем полученный файл графа
    with open("data/graph/graph_data.xlsx", "wb") as buffer:
        buffer.write(await file.read())
    
    # Загружаем обновленный граф
    G = load_graph_from_excel("data/graph/graph_data.xlsx")
    return {"message": "Graph updated successfully"}

@app.get("/hello_world")
async def hello_world():
    return {"message": "Hello, World!"}

# Новый endpoint для генерации начального расписания
@app.get("/generate_default_schedule")
async def generate_default_schedule():
    # Загружаем данные о заявках и ледоколах
    requests = pd.read_csv('data/requests/requests_fixed.csv')
    ledokoly = pd.read_csv('data/icebreakers/icebreakers.csv')
    
    # Преобразование строк с датами в объекты datetime
    requests['start_date'] = pd.to_datetime(requests['start_date'], format='%d-%m-%y %H:%M')
    ledokoly['start_date'] = pd.to_datetime(ledokoly['start_date'], format='%d-%m-%y %H:%M')

    # Преобразование в словари
    ships = requests.to_dict('records')
    icebreakers = ledokoly.to_dict('records')

    # Загрузка графа
    G = load_graph_from_excel("data/graph/graph_data.xlsx")

    # Создание начального расписания
    initial_schedule = fcfs_scheduling(ships, icebreakers, G)

    # Настройка параметров имитации отжига
    initial_temp = 1000
    cooling_rate = 0.99

    # Оптимизация расписания
    best_schedule, best_cost = simulated_annealing(initial_schedule, cost_function, initial_temp, cooling_rate, G, requests)

    # Преобразование numpy типов в стандартные Python типы
    def convert_to_json_serializable(data):
        if isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: convert_to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_json_serializable(item) for item in data]
        else:
            return data

    # Преобразование данных расписания в формат JSON
    best_schedule_serializable = convert_to_json_serializable(best_schedule)
    
    return best_schedule_serializable