# utils.py

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

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