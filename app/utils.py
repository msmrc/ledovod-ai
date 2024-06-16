# utils.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from scipy.spatial import KDTree

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

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




def haversine(coord1, coord2):
    # ...

def check_accessibility(ice_conditions_average, ice_class):
    # ...

def calculate_average_speed(ice_conditions_average, ice_class, ship_speed=20, provodka=False, icebreaker_class=10):
    # ...

def calculate_traverse_time(distance, average_speed):
    # ...

def load_graph_from_excel(file_path):
    # ...

def update_weights(G, ship_speed, ice_class, icebreaker_class=10):
    # ...

def calculate_travel_time(path):
    # ...

def find_path(G, ship_speed, ice_class, icebreaker_class, start_id_real, end_id_real):
    # ...

def assign_icebreaker_to_ship(G, icebreaker, ship):
    # ...

def fcfs_scheduling(ships, icebreakers, G):
    # ...

def generate_neighbor(schedule, G, ships_df):
    # ...

def simulated_annealing(schedule, cost_function, temp, cooling_rate, G, ships_df):
    # ...

def acceptance_probability(old_cost, new_cost, temperature):
    # ...

def cost_function(schedule):
    # ...