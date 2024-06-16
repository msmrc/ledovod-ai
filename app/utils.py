# utils.py
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import random
import math
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from scipy.spatial import KDTree
import json


"""
1. Функции для работы с ледовыми условиями на отрезке пути
"""

# Функция для преобразования широты и долготы в координаты сетки
def lat_lon_to_grid(lat, lon, lat_grid, lon_grid):
    lat_values = lat_grid.values.flatten()
    lon_values = lon_grid.values.flatten()
    lat_idx = np.abs(lat_values - lat).argmin()
    lon_idx = np.abs(lon_values - lon).argmin()
    lat_idx = np.clip(lat_idx, 0, lat_values.shape[0] - 1)
    lon_idx = np.clip(lon_idx, 0, lon_values.shape[0] - 1)
    return lat_idx, lon_idx

# Функция для расчета начального азимута (угла)
def initial_bearing(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    diff_long = np.radians(lon2 - lon1)
    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# Функция для получения пути большого круга
def get_great_circle_path(lat1, lon1, lat2, lon2, num_points=30):
    start = (lat1, lon1)
    end = (lat2, lon2)
    bearing = initial_bearing(lat1, lon1, lat2, lon2)
    path = [start]
    for i in range(1, num_points):
        fraction = i / num_points
        distance = geodesic(start, end).km * fraction
        intermediate_point = geodesic(kilometers=distance).destination(point=start, bearing=bearing)
        path.append((intermediate_point.latitude, intermediate_point.longitude))
    path.append(end)
    return path

# Функция для расчета ледовых условий вдоль пути
def calculate_ice_statistics(path, ice_data, lat_grid, lon_grid):
    lat_values = lat_grid.values.flatten()
    lon_values = lon_grid.values.flatten()
    ice_values = ice_data.values.flatten()

    # Создаем KDTree для быстрого поиска ближайшего соседа
    tree = KDTree(np.column_stack([lat_values, lon_values]))
    
    ice_conditions = []
    for lat, lon in path:
        _, idx = tree.query([lat, lon])
        ice_conditions.append(ice_values[idx])
    
    min_ice = np.min(ice_conditions)
    max_ice = np.max(ice_conditions)
    avg_ice = np.mean(ice_conditions)
    
    # Рассчитываем среднее значение для нижней трети ледовых условий
    ice_conditions_sorted = sorted(ice_conditions)
    lowest_one_third_ice = ice_conditions_sorted[:len(ice_conditions_sorted) // 3]
    avg_lowest_one_third_ice = np.mean(lowest_one_third_ice)
    
    return min_ice, max_ice, avg_ice, avg_lowest_one_third_ice, ice_conditions

# Функция для расчета только ледовых условий
def calculate_ice_statistics_only(sheet_name, file_path, start_point, end_point):
    xls = pd.ExcelFile(file_path)
    required_sheets = [xls.sheet_names[0], xls.sheet_names[1], sheet_name]
    for sheet in required_sheets:
        if sheet not in xls.sheet_names:
            raise ValueError(f"Sheet '{sheet}' not found in the Excel file.")

    lon = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    lat = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
    ice_data = pd.read_excel(xls, sheet_name=sheet_name)
    path = get_great_circle_path(start_point[0], start_point[1], end_point[0], end_point[1])

    min_ice, max_ice, avg_ice, avg_lowest_one_third_ice, ice_conditions = calculate_ice_statistics(path, ice_data, lat, lon)
    return min_ice, max_ice, avg_ice, avg_lowest_one_third_ice

# Функция для визуализации ледовых условий с путем
def visualize_ice_conditions_with_path(sheet_name, title, file_path, start_point, end_point):
    xls = pd.ExcelFile(file_path)
    required_sheets = [xls.sheet_names[0], xls.sheet_names[1], sheet_name]
    for sheet in required_sheets:
        if sheet not in xls.sheet_names:
            raise ValueError(f"Sheet '{sheet}' not found in the Excel file.")

    lon = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    lat = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
    lon_flat = lon.values.flatten()
    lat_flat = lat.values.flatten()
    ice_data = pd.read_excel(xls, sheet_name=sheet_name)
    ice_flat = ice_data.values.flatten()
    ice_flat_rounded = np.round(ice_flat)
    ice_flat_rounded[ice_flat_rounded >= 22] = 22

    # Настройка цветовой карты
    new_colors = np.zeros((256, 4))
    new_colors[:128, :3] = plt.cm.Reds(np.linspace(1, 0, 128))[:, :3]
    new_colors[128:, :3] = plt.cm.Blues(np.linspace(0, 1, 128))[:, :3]
    new_colors[:128, -1] = np.linspace(0, 1, 128)
    new_colors[128:, -1] = 1
    new_cmap = mcolors.ListedColormap(new_colors)

    path = get_great_circle_path(start_point[0], start_point[1], end_point[0], end_point[1])
    print("Точки пути большого круга:")
    for point in path:
        print(point)

    min_ice, max_ice, avg_ice, avg_lowest_one_third_ice, ice_conditions = calculate_ice_statistics(path, ice_data, lat, lon)
    path_lats, path_lons = zip(*path)

    min_lat = min(start_point[0], end_point[0]) - 2
    max_lat = max(start_point[0], end_point[0]) + 2
    min_lon = min(start_point[1], end_point[1]) - 2
    max_lon = max(start_point[1], end_point[1]) + 2

    plt.figure(figsize=(30, 20))
    m = Basemap(projection='laea', 
                lat_0=(min_lat + max_lat) / 2, lon_0=(min_lon + max_lon) / 2, 
                llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon,
                resolution='l')
    m.drawcoastlines()
    m.drawmapboundary()
    x, y = m(lon_flat, lat_flat)
    sc = m.scatter(x, y, c=ice_flat_rounded, cmap=new_cmap, marker='s', s=250, vmin=-10, vmax=22)
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, shrink=0.5)
    cbar.set_label('Ледовые условия (нормализованные)')
    plt.title(title, fontsize=24)

    x_path, y_path = m(path_lons, path_lats)
    m.plot(x_path, y_path, marker='o', linestyle='-', color='lime', markersize=5, linewidth=2, label='Путь большого круга')
    plt.legend(loc='lower right', fontsize='small')

    plt.show(block=True)

    return min_ice, max_ice, avg_ice, avg_lowest_one_third_ice


"""
2. Функции для создания графа и работы с ним
"""

# Функция для расчета расстояния между двумя координатами с использованием формулы гаверсинуса
def haversine(coord1, coord2):
    R = 3440.065  # Радиус Земли в морских милях
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# Класс для работы с графами на основе данных о точках и ребрах
class GraphMap:
    def __init__(self, points_path, edges_path, ice_file_path, date=None):
        self.points_path = points_path
        self.edges_path = edges_path
        self.ice_file_path = ice_file_path
        self.date = date
        self.g_points = None
        self.g_edges = None
        self.G = None
        self.G_undirected = None

    # Метод для загрузки данных из файлов
    def load_data(self):
        start_time = datetime.now()
        print(f"[{self.date}] Loading data...")
        self.g_points = pd.read_excel(self.points_path, sheet_name='points').iloc[:, :-2]
        self.g_edges = pd.read_excel(self.edges_path, sheet_name='edges')
        self.g_points.loc[self.g_points['longitude'] < 0, 'longitude'] += 360
        end_time = datetime.now()
        print(f"[{self.date}] Data loaded in {end_time - start_time}")

    # Метод для создания графа на основе загруженных данных
    def create_graph(self):
        start_time = datetime.now()
        print(f"[{self.date}] Creating graph...")
        self.G = nx.DiGraph()
        
        for _, row in self.g_edges.iterrows():
            start_point = self.g_points[self.g_points['point_id'] == row['start_point_id']].iloc[0]
            end_point = self.g_points[self.g_points['point_id'] == row['end_point_id']].iloc[0]
            start_coords = (start_point['latitude'], start_point['longitude'])
            end_coords = (end_point['latitude'], end_point['longitude'])

            min_ice, max_ice, avg_ice, avg_lowest_one_third_ice = calculate_ice_statistics_only(self.date, self.ice_file_path, start_coords, end_coords)
            distance = haversine(start_coords, end_coords)
            
            self.G.add_edge(
                row['start_point_id'], 
                row['end_point_id'], 
                distance=distance, 
                status=row['status'],
                min_ice=min_ice,
                max_ice=max_ice,
                avg_ice=avg_ice,
                avg_lowest_one_third_ice=avg_lowest_one_third_ice
            )

        self.G_undirected = self.G.to_undirected()
        end_time = datetime.now()
        print(f"[{self.date}] Graph created in {end_time - start_time}")

    # Метод для вычисления кратчайшего пути между двумя точками
    def calculate_shortest_path(self, source_id, target_id):
        return nx.dijkstra_path(self.G_undirected, source=source_id, target=target_id, weight='distance')

    # Метод для получения длины пути
    def get_path_length(self, path):
        return int(sum(self.G_undirected[start][end]['distance'] for start, end in zip(path[:-1], path[1:])))

    # Метод для поиска пути между двумя точками
    def find_path(self, source_id, target_id):
        start_time = datetime.now()
        self.load_data()
        self.create_graph()
        shortest_path = self.calculate_shortest_path(source_id, target_id)
        path_length = self.get_path_length(shortest_path)
        end_time = datetime.now()
        print(f"[{self.date}] Path found in {end_time - start_time}")
        return [int(point_id) for point_id in shortest_path], path_length

    # Метод для сохранения графа в файл
    def save_graph(self, date, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        nodes_data = []
        for node in self.G.nodes:
            node_data = self.g_points[self.g_points['point_id'] == node].iloc[0]
            connections = list(self.G.neighbors(node)) + [n for n in self.G.predecessors(node)]
            nodes_data.append({
                'id': node,
                'name': node_data.get('name', ''),
                'latitude': node_data['latitude'],
                'longitude': node_data['longitude'],
                'connections': connections,
                'type': ''
            })
        edges_data = []
        for start, end, data in self.G.edges(data=True):
            edges_data.append({
                'start_point_id': start,
                'end_point_id': end,
                'distance': data['distance'],
                'ice_conditions_max': data['max_ice'],
                'ice_conditions_min': data['min_ice'],
                'ice_conditions_average': data['avg_ice'],
                'ice_conditions_avg_lowest_one_third': data['avg_lowest_one_third_ice']
            })
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)

        file_path = os.path.join(directory, f'graph_data_{date}.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            nodes_df.to_excel(writer, sheet_name='nodes', index=False)
            edges_df.to_excel(writer, sheet_name='edges', index=False)

    # Метод для загрузки графа из файла
    def load_graph(self, date, directory):
        file_path = os.path.join(directory, f'graph_data_{date}.xlsx')
        nodes_df = pd.read_excel(file_path, sheet_name='nodes')
        edges_df = pd.read_excel(file_path, sheet_name='edges')
        
        self.g_points = nodes_df  # Загрузка данных о точках
        self.G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            self.G.add_edge(
                row['start_point_id'], 
                row['end_point_id'], 
                distance=row['distance'], 
                ice_conditions_max=row['ice_conditions_max'],
                ice_conditions_min=row['ice_conditions_min'],
                ice_conditions_average=row['ice_conditions_average'],
                ice_conditions_avg_lowest_one_third=row['ice_conditions_avg_lowest_one_third']
            )
        self.G_undirected = self.G.to_undirected()

# Функция для обработки данных за определенную дату
def process_date(date, points_path, edges_path, ice_file_path, directory):
    print(f"Processing {date}...")
    start_time = datetime.now()
    graph_map = GraphMap(points_path, edges_path, ice_file_path, date)
    graph_map.load_data()
    graph_map.create_graph()
    graph_map.save_graph(date, directory)
    end_time = datetime.now()
    print(f"Processing {date} completed in {end_time - start_time}")
    
    
"""
3. Правила проверки проходимости и скорости судна
"""

# Функция для проверки доступности участка пути в зависимости от условий льда и класса льда
def check_accessibility(ice_conditions_average, ice_class):
    if ice_class in [0, 1, 2, 3] and ice_conditions_average < 14:
        return 3, False  # Движение запрещено
    if ice_conditions_average < 10:
        return 3, False  # Движение запрещено
    if ice_conditions_average > 20:
        return 0, True  # Самостоятельное движение
    if 10 <= ice_conditions_average <= 14:
        if ice_class in [4, 5, 6, 7]:
            return 1, True  # Движение с проводкой ледокола
        elif ice_class in [10, 11]:
            return 0, True  # Самостоятельное движение
        else:
            return 3, False  # Движение запрещено
    if 15 <= ice_conditions_average <= 19:
        if ice_class in [0, 1, 2, 3, 4, 5, 6]:
            return 1, True  # Движение с проводкой ледокола
        elif ice_class in [7, 10, 11]:
            return 0, True  # Самостоятельное движение
        else:
            return 3, False  # Движение запрещено
    return 3, False  # Добавим дефолтное возвращение значения, если не одно из условий не выполнено

# Функция для расчета средней скорости движения на участке в зависимости от условий льда и проводки ледокола
def calculate_average_speed(ice_conditions_average, ice_class, ship_speed=20, provodka=False, icebreaker_class=10):
    """
    Функция calculate_average_speed возвращает среднюю скорость движения на участке в зависимости от условий льда и проводки ледокола.
    """
    
    if 10 <= ice_conditions_average <= 14:
        if ice_class in [4, 5, 6]:
            avg_speed = ship_speed * 0.7  # Замедление на 30%
        elif ice_class == 7:
            avg_speed = ship_speed * 0.8  # Замедление на 20%
        elif ice_class == 11:
            avg_speed = 12  # Скорость 12 узлов
        elif ice_class == 10:
            avg_speed = 9  # Скорость 9 узлов
        else:
            avg_speed = 0  # Непроходимый участок
    elif 15 <= ice_conditions_average <= 19:
        if ice_class in [0, 1, 2, 3]:
            avg_speed = ship_speed * 0.5  # Замедление на 50%
        elif ice_class in [4, 5, 6]:
            avg_speed =ship_speed * 0.8  # Замедление на 20%
        elif ice_class == 7:
            avg_speed = ship_speed * 0.6 # Замедление на 40%
        elif ice_class == 11:
            avg_speed = 17 # Скорость 17 узлов
        elif ice_class == 10:
            avg_speed =15.5  # Скорость 15.5 узлов
        else:
            avg_speed = 0  # Непроходимый участок
    elif ice_conditions_average >= 20:
        if ice_class in [0, 1, 2, 3, 4, 5, 6, 7]:
            avg_speed = ship_speed  # Без замедления
        else:
            if ice_class == 10:
                avg_speed = 18.5  # Скорость 18.5 узлов
            elif ice_class == 11:
                avg_speed = 21.5  # Скорость 21.5 узлов
    
    icebreaker_speed = avg_speed
    
    if provodka:
        if icebreaker_class == 10:
            if 19 < ice_conditions_average:
                icebreaker_speed = 18.5
            elif 15 <= ice_conditions_average <= 19:
                icebreaker_speed = 15.5   
            if 10 <= ice_conditions_average <= 14:
                icebreaker_speed = 9
            
        elif icebreaker_class == 11:
            if 19 < ice_conditions_average:
                icebreaker_speed = 21.5
            elif 15 <= ice_conditions_average <= 19:
                icebreaker_speed = 17   
            if 10 <= ice_conditions_average <= 14:
                icebreaker_speed = 12

    return round(min(avg_speed, icebreaker_speed, 1))

def calculate_traverse_time(distance, average_speed):
    if average_speed == 0:
        return 0
    return round(distance / average_speed, 2)


"""
4. Функции для работы с загрузкой обработанного графа и формула пересчёта весов рёбер под конкретное судно
"""

# Функция для загрузки графа из файла Excel
def load_graph_from_excel(file_path):
    # Загрузка данных из файла
    nodes_df = pd.read_excel(file_path, sheet_name='nodes')
    edges_df = pd.read_excel(file_path, sheet_name='edges')

    # Создание пустого графа
    G = nx.Graph()

    # Добавление узлов в граф
    for _, row in nodes_df.iterrows():
        G.add_node(row['id'], name=row['name'], latitude=row['latitude'], longitude=row['longitude'], connections=row['connections'], type=row['type'])

    # Добавление рёбер в граф
    for _, row in edges_df.iterrows():
        ice_conditions_avg = max(row['ice_conditions_average'], 10)
        G.add_edge(
            row['start_point_id'], 
            row['end_point_id'], 
            distance=round(row['distance']), 
            ice_conditions_max=round(row['ice_conditions_max']),
            ice_conditions_min=round(row['ice_conditions_min']),
            ice_conditions_average=round(ice_conditions_avg),
            ice_conditions_avg_lowest_one_third=round(row['ice_conditions_avg_lowest_one_third'])
        )

    return G

# Функция для обновления весов рёбер графа под конкретное судно
def update_weights(G, ship_speed, ice_class, icebreaker_class=10):
    for u, v, data in G.edges(data=True):
        ice_conditions_avg = data['ice_conditions_average']
        provodka_needed, _ = check_accessibility(ice_conditions_avg, ice_class)
        average_speed = calculate_average_speed(ice_conditions_avg, ice_class, ship_speed, provodka_needed, icebreaker_class)
        traverse_time = calculate_traverse_time(data['distance'], average_speed)
        
        data['weight'] = round(traverse_time, 2)
        data['provodka_needed'] = provodka_needed

    
"""
5. Функция для поиска кратчайшего пути на основе алгоритма Дейкстры
"""
   
# Функция для поиска пути и расчета дополнительных данных
def find_path(G, ship_speed, ice_class, icebreaker_class, start_id_real, end_id_real):
    # Обновление весов рёбер
    update_weights(G, ship_speed, ice_class, icebreaker_class)
    # Расчет кратчайшего пути
    path = nx.dijkstra_path(G, start_id_real, end_id_real, weight='weight')
    
    # Создаем список переходов с дополнительной информацией
    detailed_path = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = G[u][v]
        transition = {
            'from': u,
            'to': v,
            'distance': data['distance'],
            'ice_conditions_average': data['ice_conditions_average'],
            'average_speed': ship_speed,  # средняя скорость
            'time': round(data['distance'] / ship_speed, 2),  # время прохождения
            'icebreaker_needed': data['provodka_needed'] # необходимость в ледоколе
        }
        detailed_path.append(transition)
    return detailed_path

"""
6. Функции для нахождения изначального решения по алгоритму FCFS
"""

# Функция для назначения ледокола кораблю по условиям нахождения корабля который может приплыть как можно быстрее
def assign_icebreaker_to_ship(G, icebreaker, ship):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    
    icebreaker_start_time = icebreaker['start_date']
    icebreaker_position = icebreaker['start_id_real']
    path_to_ship = find_path(G, icebreaker['speed'], icebreaker['ice_class_int'], icebreaker['ice_class_int'], icebreaker_position, ship['start_id_real'])
    time_to_ship = calculate_travel_time(path_to_ship)
    arrival_time = icebreaker_start_time + timedelta(hours=time_to_ship)
    ship_start_time = ship['start_date']
    
    if arrival_time < ship_start_time:
        icebreaker_waiting_for_ship = round((ship_start_time - arrival_time).total_seconds() / 3600.0)
        ship_waiting_time = 0
    else:
        icebreaker_waiting_for_ship = 0
        ship_waiting_time = round((arrival_time - ship_start_time).total_seconds() / 3600.0)
    
    path_to_destination = find_path(G, icebreaker['speed'], icebreaker['ice_class_int'], icebreaker['ice_class_int'], ship['start_id_real'], ship['end_id_real'])
    time_to_destination = calculate_travel_time(path_to_destination)
    icebreaker_position = ship['end_id_real']
    icebreaker_available_time = ship_start_time + timedelta(hours=ship_waiting_time + time_to_ship + time_to_destination)
    
    # Format the datetimes
    arrival_time = arrival_time.strftime(datetime_format)
    icebreaker_available_time = icebreaker_available_time.strftime(datetime_format)
    
    return {
        'icebreaker_path': path_to_ship,
        'icebreaker_waiting_for_ship': icebreaker_waiting_for_ship,
        'ship_waiting_time': ship_waiting_time,
        'ship_path': path_to_destination,
        'icebreaker_available_time': icebreaker_available_time,
        'icebreaker_available_position': icebreaker_position,
        'time_to_ship': time_to_ship,
        'time_to_destination': time_to_destination,
        'arrival_time': arrival_time
    }

# Функция для принятия решения о назначении ледокола кораблям по логике какой ледокол ближе к кораблю тот и назначается
def fcfs_scheduling(ships, icebreakers, G):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    ships.sort(key=lambda x: x['start_date'])
    icebreaker_schedule = []
    ship_schedule = []
    total_ship_waiting_time = 0
    total_icebreaker_waiting_time = 0
    total_icebreaker_travel_time = 0
    total_ship_travel_time = 0
    
    for ship in ships:
        best_icebreaker = None
        best_result = None
        
        for icebreaker in icebreakers:
            result = assign_icebreaker_to_ship(G, icebreaker, ship)
            if best_result is None or result['ship_waiting_time'] < best_result['ship_waiting_time']:
                best_icebreaker = icebreaker
                best_result = result
        
        if best_icebreaker and best_result:
            best_icebreaker['start_id_real'] = best_result['icebreaker_available_position']
            best_icebreaker['start_date'] = pd.to_datetime(best_result['icebreaker_available_time'], format=datetime_format)
            
            icebreaker_schedule.append({
                'icebreaker_id': best_icebreaker['ship_id'],
                'icebreaker_name': best_icebreaker['ship_name'],
                'ship_id': ship['ship_id'],
                'ship_name': ship['ship_name'],
                'start_time': ship['start_date'].strftime(datetime_format),
                'arrival_time': best_result['arrival_time'],
                'end_time': best_result['icebreaker_available_time'],
                'wait_time_hours': best_result['icebreaker_waiting_for_ship'],
                'path': best_result['icebreaker_path'] + best_result['ship_path']
            })
            
            ship_schedule.append({
                'ship_id': ship['ship_id'],
                'ship_name': ship['ship_name'],
                'start_time': ship['start_date'].strftime(datetime_format),
                'end_time': best_result['icebreaker_available_time'],
                'wait_time_hours': best_result['ship_waiting_time'],
                'path': best_result['ship_path']
            })
            
            total_ship_waiting_time += best_result['ship_waiting_time']
            total_icebreaker_waiting_time += best_result['icebreaker_waiting_for_ship']
            total_icebreaker_travel_time += best_result['time_to_ship'] + best_result['time_to_destination']
            total_ship_travel_time += best_result['time_to_destination']
    
    summary = {
        'total_ship_waiting_time': round(total_ship_waiting_time),
        'total_icebreaker_waiting_time': round(total_icebreaker_waiting_time),
        'total_icebreaker_travel_time': round(total_icebreaker_travel_time),
        'total_ship_travel_time': round(total_ship_travel_time)
    }
    
    return {
        'icebreaker_schedule': icebreaker_schedule,
        'ship_schedule': ship_schedule,
        'summary': summary
    }

"""
7. Функции для алгоритма имитации отжига для оптимизации расписания
"""

# Функция для расчета общего времени для пути
def calculate_travel_time(path):
    total_time = 0
    for segment in path:
        total_time += segment['time']
    return round(total_time)

# Функция для генерации соседнего решения
def generate_neighbor(schedule, G, ships_df):
    new_schedule = {
        'icebreaker_schedule': [icebreaker.copy() for icebreaker in schedule['icebreaker_schedule']],
        'ship_schedule': [ship.copy() for ship in schedule['ship_schedule']]
    }
    ship1, ship2 = random.sample(range(len(new_schedule['ship_schedule'])), 2)
    
    # Меняем местами задания ледоколов для этих двух кораблей
    new_schedule['ship_schedule'][ship1], new_schedule['ship_schedule'][ship2] = \
        new_schedule['ship_schedule'][ship2], new_schedule['ship_schedule'][ship1]

    # Пересчитываем пути и время ожидания для затронутых кораблей
    for ship_index in [ship1, ship2]:
        icebreaker = new_schedule['icebreaker_schedule'][ship_index]
        ship = new_schedule['ship_schedule'][ship_index]

        # Проверка наличия ключей в данных корабля и заполнение их из ships_df
        required_keys = ['start_id_real', 'end_id_real', 'ship_id', 'speed', 'start_time']
        for key in required_keys:
            if key not in ship:
                if ship['ship_id'] in ships_df['ship_id'].values:
                    ship_data = ships_df[ships_df['ship_id'] == ship['ship_id']].iloc[0]
                    ship[key] = ship_data[key]
                else:
                    raise KeyError(f"Missing key in ship data: {ship}, key: {key}")

        # Определение скорости ледокола и его класса в зависимости от icebreaker_id
        if icebreaker['icebreaker_id'] == 1:
            icebreaker_speed = 22.0
            icebreaker_class = 10
        elif icebreaker['icebreaker_id'] == 2:
            icebreaker_speed = 21.0
            icebreaker_class = 10
        elif icebreaker['icebreaker_id'] == 3:
            icebreaker_speed = 18.5
            icebreaker_class = 11
        elif icebreaker['icebreaker_id'] == 4:
            icebreaker_speed = 18.5
            icebreaker_class = 11
        else:
            raise KeyError(f"Unknown icebreaker_id: {icebreaker['icebreaker_id']}")

        # Получение скорости корабля из датафрейма requests
        ship_speed = ships_df.loc[ships_df['ship_id'] == ship['ship_id'], 'speed'].values[0]

        # Пересчитываем путь и время до пункта назначения
        path_to_destination = find_path(G, ship_speed, icebreaker_class, icebreaker_class, ship['start_id_real'], ship['end_id_real'])
        time_to_destination = calculate_travel_time(path_to_destination)
        
        # Обновляем время прибытия и время ожидания
        ship_start_time = pd.to_datetime(ship['start_time'])
        arrival_time = pd.to_datetime(icebreaker['start_time']) + timedelta(hours=time_to_destination)

        if arrival_time < ship_start_time:
            icebreaker_waiting_for_ship = round((ship_start_time - arrival_time).total_seconds() / 3600.0)
            ship_waiting_time = 0
        else:
            icebreaker_waiting_for_ship = 0
            ship_waiting_time = round((arrival_time - ship_start_time).total_seconds() / 3600.0)
        
        # Обновляем информацию в расписании ледокола
        icebreaker_available_time = ship_start_time + timedelta(hours=ship_waiting_time + time_to_destination)
        icebreaker_position = ship['end_id_real']
        
        new_schedule['icebreaker_schedule'][ship_index] = {
            'icebreaker_id': icebreaker['icebreaker_id'],
            'icebreaker_name': icebreaker['icebreaker_name'],
            'ship_id': ship['ship_id'],
            'ship_name': ship['ship_name'],
            'start_time': ship['start_time'],
            'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': icebreaker_available_time.strftime('%Y-%m-%d %H:%M:%S'),
            'wait_time_hours': icebreaker_waiting_for_ship,
            'path': path_to_destination,
            'speed': icebreaker_speed,
            'ice_class_int': icebreaker_class
        }

        # Обновляем информацию в расписании корабля
        new_schedule['ship_schedule'][ship_index] = {
            'ship_id': ship['ship_id'],
            'ship_name': ship['ship_name'],
            'start_time': ship['start_time'],
            'end_time': icebreaker_available_time.strftime('%Y-%m-%d %H:%M:%S'),
            'wait_time_hours': ship_waiting_time,
            'path': path_to_destination,
            'start_id_real': ship['start_id_real'],
            'end_id_real': ship['end_id_real'],
            'speed': ship['speed']
        }

    return new_schedule

# Имитация отжига
def simulated_annealing(schedule, cost_function, temp, cooling_rate, G, ships_df):
    current_schedule = schedule
    current_cost = cost_function(current_schedule)
    best_schedule = current_schedule
    best_cost = current_cost

    while temp > 1:
        new_schedule = generate_neighbor(current_schedule, G, ships_df)
        new_cost = cost_function(new_schedule)

        if acceptance_probability(current_cost, new_cost, temp) > random.random():
            current_schedule = new_schedule
            current_cost = new_cost

        if new_cost < best_cost:
            best_schedule = new_schedule
            best_cost = new_cost

        temp *= cooling_rate

    return best_schedule, best_cost

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    return math.exp((old_cost - new_cost) / temperature)

def cost_function(schedule):
    total_ship_waiting_time = 0
    for entry in schedule['ship_schedule']:
        total_ship_waiting_time += entry['wait_time_hours']
    return total_ship_waiting_time

"""
9. Функции для построения графов с упрощенным выводом и проверками 
"""

# Функция для поиска пути и времени между двумя точками
def get_path_and_time(G, speed, ice_class, start_id, end_id):
    path = find_path(G, speed, ice_class, ice_class, start_id, end_id)
    time = calculate_travel_time(path)
    return path, time

# Функция для расчета общего времени для пути
def calculate_travel_time(path):
    total_time = 0
    for segment in path:
        total_time += segment['time']
    return round(total_time)

def assign_icebreaker_to_ship(G, icebreaker, ship):
    datetime_format = '%d-%m-%y %H:%M'

    # Время доступности ледокола после завершения текущего задания
    icebreaker_availability_time = icebreaker['availability_time']
    icebreaker_position = icebreaker['start_id_real']
    
    # Время доплытия от текущей позиции ледокола до отправной точки корабля
    path_to_ship, time_to_ship = get_path_and_time(G, icebreaker['speed'], icebreaker['ice_class_int'], icebreaker_position, ship['start_id_real'])
    arrival_time = icebreaker_availability_time + timedelta(hours=time_to_ship)
    
    # Время ожидания ледокола до прибытия к кораблю
    ship_start_time = ship['start_date']
    if arrival_time < ship_start_time:
        icebreaker_waiting_for_ship = round((ship_start_time - arrival_time).total_seconds() / 3600.0)
        ship_waiting_time = 0
    else:
        icebreaker_waiting_for_ship = 0
        ship_waiting_time = round((arrival_time - ship_start_time).total_seconds() / 3600.0)
    
    # Время пути ледокола вместе с кораблем
    path_with_ship, time_to_destination = get_path_and_time(G, ship['speed'], ship['ice_class_int'], ship['start_id_real'], ship['end_id_real'])
    icebreaker_position_after_delivery = ship['end_id_real']
    
    # Время окончания пути вместе с ледоколом
    end_time_with_ship = ship_start_time + timedelta(hours=ship_waiting_time + time_to_ship + time_to_destination)
    
    icebreaker_info = {
        'initial_position': icebreaker_position,  # Начальная позиция ледокола перед началом текущего задания
        'path_to_ship': path_to_ship,  # Путь ледокола до корабля, который нуждается в проводке
        'time_to_ship': time_to_ship,  # Время, необходимое ледоколу, чтобы доплыть до корабля
        'waiting_for_ship': icebreaker_waiting_for_ship,  # Время ожидания ледокола перед началом проводки корабля
        'path_with_ship': path_with_ship,  # Путь ледокола вместе с кораблем
        'start_time_with_ship': ship_start_time.strftime(datetime_format),  # Время начала проводки корабля ледоколом
        'time_to_destination': time_to_destination,  # Время, необходимое для проводки корабля до конечной точки
        'arrival_time': arrival_time.strftime(datetime_format),  # Время прибытия ледокола к кораблю
        'end_time_with_ship': end_time_with_ship.strftime(datetime_format),  # Время окончания проводки корабля ледоколом
        'position_after_successful_delivery': icebreaker_position_after_delivery  # Позиция ледокола после завершения проводки корабля
    }

    ship_info = {
        'ship_waiting_time': ship_waiting_time,  # Время ожидания корабля до прибытия ледокола
        'start_time_with_icebreaker': ship_start_time.strftime(datetime_format),  # Время начала проводки корабля ледоколом
        'ship_path': path_with_ship,  # Путь корабля вместе с ледоколом
        'ship_path_time': time_to_destination,  # Время, необходимое для проводки корабля до конечной точки
        'end_time_with_icebreaker': end_time_with_ship.strftime(datetime_format)  # Время окончания проводки корабля ледоколом
    }



    return icebreaker_info, ship_info

def fcfs_scheduling2(ships, icebreakers, G):
    datetime_format = '%d-%m-%y %H:%M'
    
    # Сортируем корабли по времени начала, чтобы обработать их в порядке их готовности к отправке
    ships.sort(key=lambda x: x['start_date'])
    
    icebreaker_schedule = []  # Расписание для ледоколов
    ship_schedule = []  # Расписание для кораблей
    total_times = initialize_total_times()  # Инициализируем общие временные показатели

    for ship in ships:
        # Находим лучший ледокол для текущего корабля
        best_icebreaker, best_result = find_best_icebreaker(G, icebreakers, ship)

        if best_icebreaker and best_result:

            # Проверяем, что ледокол не занят другим заданием на момент начала этого задания
            if best_icebreaker['availability_time'] <= ship['start_date']:
                # Обновляем информацию о ледоколе после назначения кораблю
                update_icebreaker(best_icebreaker, best_result[0], datetime_format)
                
                # Обновляем расписания ледоколов и кораблей
                update_schedules(icebreaker_schedule, ship_schedule, best_icebreaker, ship, best_result, datetime_format)
                
                # Обновляем общие временные показатели
                update_total_times(total_times, best_result)
            else:

                # Если ледокол не доступен, добавляем время ожидания к доступности ледокола
                best_icebreaker['availability_time'] += timedelta(hours=best_result[0]['time_to_ship'])
                update_icebreaker(best_icebreaker, best_result[0], datetime_format)
                update_schedules(icebreaker_schedule, ship_schedule, best_icebreaker, ship, best_result, datetime_format)
                update_total_times(total_times, best_result)


    # Сводка общих временных затрат
    summary = summarize_total_times(total_times)
    
    # Возвращаем расписание ледоколов, кораблей и сводку
    return {
        'icebreaker_schedule': icebreaker_schedule,
        'ship_schedule': ship_schedule,
        'summary': summary
    }

def initialize_total_times():
    # Инициализируем словарь для хранения общих временных затрат
    return {
        'total_ship_waiting_time': 0,
        'total_icebreaker_waiting_time': 0,
        'total_icebreaker_travel_time': 0,
        'total_ship_travel_time': 0
    }

def find_best_icebreaker(G, icebreakers, ship):
    best_icebreaker = None
    best_result = None

    for icebreaker in icebreakers:
        result = assign_icebreaker_to_ship(G, icebreaker, ship)

        # Выбираем ледокол, который минимизирует время ожидания корабля
        if not best_result or result[1]['ship_waiting_time'] < best_result[1]['ship_waiting_time']:
            best_icebreaker = icebreaker
            best_result = result

    return best_icebreaker, best_result

def update_icebreaker(icebreaker, result, datetime_format):
    # Обновляем позицию ледокола после завершения текущего задания
    icebreaker['start_id_real'] = result['position_after_successful_delivery']
    # Обновляем время доступности ледокола после завершения текущего задания
    icebreaker['availability_time'] = pd.to_datetime(result['end_time_with_ship'], format=datetime_format)

def update_schedules(icebreaker_schedule, ship_schedule, icebreaker, ship, result, datetime_format):
    icebreaker_info, ship_info = result
    
    # Обновляем расписание для ледокола
    icebreaker_schedule.append({
        'icebreaker_id': icebreaker['ship_id'],
        'icebreaker_name': icebreaker['ship_name'],
        'initial_position': icebreaker_info['initial_position'],  # Начальная позиция ледокола
        'path_to_ship': icebreaker_info['path_to_ship'],  # Путь ледокола до корабля
        'time_to_ship': icebreaker_info['time_to_ship'],  # Время пути до корабля
        'waiting_for_ship': icebreaker_info['waiting_for_ship'],  # Время ожидания корабля
        'path_with_ship': icebreaker_info['path_with_ship'],  # Путь с кораблем
        'start_time_with_ship': icebreaker_info['start_time_with_ship'],  # Время начала пути с кораблем
        'time_to_destination': icebreaker_info['time_to_destination'],  # Время пути до пункта назначения
        'arrival_time': icebreaker_info['arrival_time'],  # Время прибытия к кораблю
        'end_time_with_ship': icebreaker_info['end_time_with_ship'],  # Время окончания пути с кораблем
        'position_after_successful_delivery': icebreaker_info['position_after_successful_delivery'],  # Позиция после доставки
        'ship_id': ship['ship_id'],
        'ship_name': ship['ship_name']
    })
    
    # Обновляем расписание для корабля
    ship_schedule.append({
        'ship_id': ship['ship_id'],
        'ship_name': ship['ship_name'],
        'ship_waiting_time': ship_info['ship_waiting_time'],  # Время ожидания корабля
        'start_time_with_icebreaker': ship_info['start_time_with_icebreaker'],  # Время начала пути с ледоколом
        'ship_path': ship_info['ship_path'],  # Путь корабля с ледоколом
        'ship_path_time': ship_info['ship_path_time'],  # Время пути корабля с ледоколом
        'end_time_with_icebreaker': ship_info['end_time_with_icebreaker']  # Время окончания пути с ледоколом
    })


def update_total_times(total_times, result):
    icebreaker_info, ship_info = result  # Декомпозиция result на icebreaker_info и ship_info
    # Обновляем общие временные затраты на основе текущего задания
    total_times['total_ship_waiting_time'] += ship_info['ship_waiting_time']
    total_times['total_icebreaker_waiting_time'] += icebreaker_info.get('waiting_for_ship', 0)
    total_times['total_icebreaker_travel_time'] += icebreaker_info['time_to_ship'] + icebreaker_info['time_to_destination']
    total_times['total_ship_travel_time'] += ship_info['ship_path_time']



def summarize_total_times(total_times):
    # Сводка общих временных затрат с округлением до целых значений
    return {
        'total_ship_waiting_time': round(total_times['total_ship_waiting_time']),
        'total_icebreaker_waiting_time': round(total_times['total_icebreaker_waiting_time']),
        'total_icebreaker_travel_time': round(total_times['total_icebreaker_travel_time']),
        'total_ship_travel_time': round(total_times['total_ship_travel_time'])
    }


# Функция для преобразования данных в нужный формат JSON
def transform_scheduling_result(scheduling_result, filename):
    transformed_data = []

    def format_entry(name, id, ship_type, travel_type, initial_position, final_position, departure_time, arrival_time):
        return {
            "name": name,
            "id": id,
            "ship_type": ship_type,
            "travel_type": travel_type,
            "initial_position": initial_position,
            "final_position": final_position,
            "departure_time": departure_time,
            "arrival_time": arrival_time
        }
    
    # Обрабатываем расписание ледоколов
    for entry in scheduling_result['icebreaker_schedule']:
        transformed_data.append(format_entry(
            entry['icebreaker_name'],
            entry['icebreaker_id'],
            'icebreaker',
            'self',
            entry['initial_position'],
            entry['path_to_ship'][0]['from'] if entry['path_to_ship'] else entry['initial_position'],
            entry['start_time_with_ship'],
            entry['arrival_time']
        ))
        transformed_data.append(format_entry(
            entry['icebreaker_name'],
            entry['icebreaker_id'],
            'icebreaker',
            'with ship',
            entry['path_to_ship'][-1]['to'] if entry['path_to_ship'] else entry['initial_position'],
            entry['position_after_successful_delivery'],
            entry['start_time_with_ship'],
            entry['end_time_with_ship']
        ))

    # Обрабатываем расписание кораблей
    for entry in scheduling_result['ship_schedule']:
        transformed_data.append(format_entry(
            entry['ship_name'],
            entry['ship_id'],
            'ship',
            'self',
            entry['ship_path'][0]['from'] if entry['ship_path'] else entry['ship_id'],
            entry['ship_path'][0]['from'] if entry['ship_path'] else entry['ship_id'],
            entry['start_time_with_icebreaker'],
            entry['start_time_with_icebreaker']
        ))
        transformed_data.append(format_entry(
            entry['ship_name'],
            entry['ship_id'],
            'ship',
            'with icebreaker',
            entry['ship_path'][0]['from'] if entry['ship_path'] else entry['ship_id'],
            entry['ship_path'][-1]['to'] if entry['ship_path'] else entry['ship_id'],
            entry['start_time_with_icebreaker'],
            entry['end_time_with_icebreaker']
        ))

    return transformed_data