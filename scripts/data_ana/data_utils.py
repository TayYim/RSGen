import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from pathlib import Path
import pickle

def find_match_from_seach_collector(l, search_df):
    l = [round(i, 4) for i in l]

    var_names = search_df.columns[7:].to_list()

    for var_name in var_names:
        search_df[var_name] = search_df[var_name].round(4)

    params_columns = var_names  # Get the names of all parameter columns
    filter_condition = (search_df[params_columns] == l).all(axis=1)

    ## Use the filter condition to filter the data
    filtered_data = search_df[filter_condition]

    match = filtered_data.iloc[0]

    ttc = match['ttc']
    loss = match['loss']
    # print(f"Match found: TTC = {ttc}, Loss = {loss}")

    return match


def get_unique_search_df(exp_path):
    # get the filename of the search data, start with search_ , end with .csv
    search_files = [f for f in os.listdir(exp_path) if f.startswith('search_') and f.endswith('.csv')]
    search_df = pd.read_csv(f"{exp_path}/{search_files[0]}")

    search_df_unique = search_df.drop_duplicates(subset=search_df.columns[7:], keep='first')
    return search_df_unique

def calculate_time_diff(row, next_row):
    try:
        current_timestamp = float(row['search_id'].split('-')[0])
        next_timestamp = float(next_row['search_id'].split('-')[0])
        return next_timestamp - current_timestamp
    except:
        return None

def get_time_diff_df(this_df):
    this_df = this_df.copy()
    time_diffs = []
    for i in range(len(this_df) - 1):
        time_diff = calculate_time_diff(this_df.iloc[i], this_df.iloc[i+1])
        time_diffs.append(time_diff)
    time_diffs.append(time_diffs[-1])
    this_df['time_consumed'] = time_diffs
    return this_df