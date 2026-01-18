# modules/data_handler.py
import pandas as pd

def load_dataset():
    data_sales = pd.read_csv('data/sales.csv')
    data_production = pd.read_csv('data/production.csv')
    data_quality = pd.read_csv('data/quality.csv')
    data_purchasing = pd.read_csv('data/purchasing.csv')
    data_inventory = pd.read_csv('data/inventory.csv')

    table = pd.DataFrame({'Counts': [data_sales.shape[0], data_sales.shape[1]]}, index=['Rows', 'Columns'])
    return data_sales, data_production, data_quality, data_purchasing, data_inventory
