# main.py

from fastapi import FastAPI, UploadFile, File
from app.utils import *

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

@app.post("/schedule")
async def generate_schedule(file: UploadFile = File(...)):
    # Сохраняем полученный файл заявок
    with open("data/requests/requests_fixed.csv", "wb") as buffer:
        buffer.write(await file.read())
    
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
    cooling_rate = 0.9999

    # Оптимизация расписания
    best_schedule, best_cost = simulated_annealing(initial_schedule, cost_function, initial_temp, cooling_rate, G, requests)

    return best_schedule 