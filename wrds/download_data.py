"""
Data Downloading Module for WRDS IBES Earnings Data

This module handles downloading and processing earnings forecast data from WRDS IBES database.
It includes functions for:
- Connecting to WRDS
- Downloading IBES data for US companies
- Aligning dates across companies using linear assignment
- Organizing data into quarterly structures
"""

import wrds
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings("ignore")


def connect_wrds(wrds_username):
    """
    Connect to WRDS database.
    
    Args:
        wrds_username (str): WRDS username
        
    Returns:
        wrds.Connection: WRDS connection object
    """
    return wrds.Connection(wrds_username=wrds_username)


def load_tickers_from_file(filename):
    """
    Load ticker symbols from a text file.
    
    Args:
        filename (str): Path to text file with one ticker per line
        
    Returns:
        list: List of ticker symbols
    """
    tickers = []
    with open(filename, 'r') as f:
        for line in f:
            tickers.append(line.strip())
    return tickers


def download_ibes_tickers(conn, us_tickers, start_date='01/01/2010', end_date='01/01/2025'):
    """
    Download IBES ticker mappings for US companies.
    
    Args:
        conn: WRDS connection object
        us_tickers (list): List of US ticker symbols
        start_date (str): Start date for data query (default: '01/01/2010')
        end_date (str): End date for data query (default: '01/01/2025')
        
    Returns:
        list: List of tuples (oftic, ibes_ticker) for companies with IBES data
    """
    ibes_tickers = []
    for t in tqdm(us_tickers):
        try:
            data = conn.raw_sql(f"""SELECT ticker, oftic, cname, estimator, analys, FPI, MEASURE, VALUE, FPEDATS, ANNDATS, ANNTIMS, ACTUAL, ANNDATS_ACT, ANNTIMS_ACT
                             FROM tr_ibes.det_epsus
                             WHERE oftic = '{t}'
                             and anndats >= '{start_date}'
                             and fpi = '6'
                             """)
            if data.shape[0] == 0:
                print(f"No data for {t}")
            else:
                ibes_ticker = data['ticker'].iloc[0]
                ibes_tickers.append((t, ibes_ticker))
        except Exception as e:
            print(f"Error for {t}: {e}")
    return ibes_tickers


def download_ibes_data(conn, ibes_tickers, output_dir='data/ibes', start_date='01/01/2010', end_date='01/01/2025'):
    """
    Download IBES earnings data for each ticker and save to CSV files.
    
    Args:
        conn: WRDS connection object
        ibes_tickers (list): List of tuples (oftic, ibes_ticker)
        output_dir (str): Directory to save CSV files (default: 'data/ibes')
        start_date (str): Start date for data query (default: '01/01/2010')
        end_date (str): End date for data query (default: '01/01/2025')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for oftic, ibes in tqdm(ibes_tickers):
        data = conn.raw_sql(f"""SELECT ticker, oftic, cname, estimator, analys, FPI, MEASURE, VALUE, FPEDATS, ANNDATS, ANNTIMS, ACTUAL, ANNDATS_ACT, ANNTIMS_ACT
                         FROM tr_ibes.det_epsus
                         WHERE ticker = '{ibes}'
                         and anndats >= '{start_date}'
                         and anndats <= '{end_date}'
                         and fpi = '6'
                         """)
        data.to_csv(f"{output_dir}/{oftic}.csv", index=False)


def load_ibes_data(data_dir='data/ibes', ibes_tickers=None):
    """
    Load IBES data from CSV files into a dictionary.
    
    Args:
        data_dir (str): Directory containing CSV files (default: 'data/ibes')
        ibes_tickers (list): Optional list of tuples (oftic, ibes_ticker) to load
        
    Returns:
        dict: Dictionary mapping ticker symbols to DataFrames
    """
    ibes_data = dict()
    if ibes_tickers is None:
        import os
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in files]
    else:
        tickers = [oftic for oftic, _ in ibes_tickers]
    
    for oftic in tqdm(tickers):
        try:
            data = pd.read_csv(f"{data_dir}/{oftic}.csv")
            ibes_data[oftic] = data
        except FileNotFoundError:
            print(f"File not found for {oftic}")
            continue
    
    return ibes_data


def align_dates(ibes_data, base_ticker='AAPL', max_length=60):
    """
    Align announcement dates across companies using linear assignment.
    Uses a base ticker (default: AAPL) as reference and aligns all other tickers to it.
    
    Args:
        ibes_data (dict): Dictionary mapping tickers to DataFrames
        base_ticker (str): Ticker to use as base for alignment (default: 'AAPL')
        max_length (int): Maximum number of dates to consider (default: 60)
        
    Returns:
        dict: Dictionary mapping tickers to aligned date arrays
    """
    unique_dates = dict()
    
    # Extract unique dates for each ticker
    for t, data in ibes_data.items():
        dates = pd.to_datetime(data['anndats_act'].dropna().unique(), format='%Y-%m-%d')
        dates = np.array(list(dates) + [pd.NaT] * (max_length - len(dates)))
        unique_dates[t] = dates
    
    # Use base ticker as reference
    base_date = unique_dates[base_ticker]
    aligned_dates = dict()
    
    # Align each ticker to base ticker using linear assignment
    for t, d in unique_dates.items():
        cost_matrix = []
        for d1 in base_date:
            row = []
            for d2 in d:
                row.append(abs((d1 - d2).days) if pd.notna(d1) and pd.notna(d2) else 365*5*2)
            cost_matrix.append(row)
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
        aligned_dates[t] = d[col_ind]
    
    return aligned_dates


def create_quarterly_data(ibes_data, aligned_dates, starting_year=2010):
    """
    Organize IBES data into quarterly structures.
    
    Args:
        ibes_data (dict): Dictionary mapping tickers to DataFrames
        aligned_dates (dict): Dictionary mapping tickers to aligned date arrays
        starting_year (int): Starting year for quarterly organization (default: 2010)
        
    Returns:
        tuple: (quarterly_data, quarterly_actual, quarterly_means)
            - quarterly_data: dict mapping (ticker, year, quarter) to DataFrame with forecasts
            - quarterly_actual: dict mapping (ticker, year, quarter) to (actual_value, date, time)
            - quarterly_means: dict mapping (ticker, year, quarter) to mean forecast value
    """
    quarterly_data = dict()
    quarterly_actual = dict()
    quarterly_means = dict()
    
    for oftic, data in tqdm(ibes_data.items()):
        data = data.dropna(axis=0, subset=['anndats_act'])
        
        for i, date in enumerate(aligned_dates[oftic]):
            quarter_num = (i % 4) + 1  # 1,2,3,4
            year = starting_year + (i // 4)
            
            if date is pd.NaT:
                quarterly_actual[oftic, year, quarter_num] = None
                quarterly_data[oftic, year, quarter_num] = None
                continue
            
            date = date.strftime('%Y-%m-%d')
            subdata = data[data['anndats_act'] == date].copy()
            
            if subdata.shape[0] == 0:
                quarterly_actual[oftic, year, quarter_num] = None
                quarterly_data[oftic, year, quarter_num] = None
                continue
            
            quarterly_actual[oftic, year, quarter_num] = (
                subdata['actual'].iloc[0], 
                subdata['anndats_act'].iloc[0],
                subdata['anntims_act'].iloc[0]
            )
            
            subdata['ann_datetime'] = pd.to_datetime(
                subdata['anndats'] + ' ' + subdata['anntims'], 
                format='%Y-%m-%d %H:%M:%S'
            )
            
            mean_value = subdata['value'].mean()
            quarterly_means[oftic, year, quarter_num] = mean_value
            
            quarterly_data[oftic, year, quarter_num] = subdata[['value', 'ann_datetime']]
    
    return quarterly_data, quarterly_actual, quarterly_means


def save_quarterly_data(quarterly_data, quarterly_actual, quarterly_means, 
                       data_prefix='quarterly'):
    """
    Save quarterly data structures to pickle files.
    
    Args:
        quarterly_data (dict): Quarterly forecast data
        quarterly_actual (dict): Quarterly actual values
        quarterly_means (dict): Quarterly mean forecasts
        data_prefix (str): Prefix for output files (default: 'quarterly')
    """
    with open(f'{data_prefix}_data.pkl', 'wb') as f:
        pickle.dump(quarterly_data, f)
    
    with open(f'{data_prefix}_actual.pkl', 'wb') as f:
        pickle.dump(quarterly_actual, f)
    
    with open(f'{data_prefix}_means.pkl', 'wb') as f:
        pickle.dump(quarterly_means, f)


def load_quarterly_data(data_prefix='quarterly'):
    """
    Load quarterly data structures from pickle files.
    
    Args:
        data_prefix (str): Prefix for input files (default: 'quarterly')
        
    Returns:
        tuple: (quarterly_data, quarterly_actual, quarterly_means)
    """
    with open(f'{data_prefix}_data.pkl', 'rb') as f:
        quarterly_data = pickle.load(f)
    
    with open(f'{data_prefix}_actual.pkl', 'rb') as f:
        quarterly_actual = pickle.load(f)
    
    with open(f'{data_prefix}_means.pkl', 'rb') as f:
        quarterly_means = pickle.load(f)
    
    return quarterly_data, quarterly_actual, quarterly_means


def get_rows_cols(user_user=True, tickers=None):
    """
    Get row and column indices for the data matrix.
    
    Args:
        user_user (bool): If True, rows are tickers and cols are (year, quarter) tuples.
                         If False, rows are (year, quarter) and cols are tickers.
        tickers (list): Optional list of ticker symbols. If None, returns placeholder.
    
    Returns:
        tuple: (rows, cols) where rows and cols are lists
    """
    cols = [(year, quarter) for year in range(2010, 2025) for quarter in range(1, 5)]
    
    if user_user is True:
        rows = tickers if tickers is not None else []
        return rows, cols
    else:
        rows = cols
        cols = tickers if tickers is not None else []
        return rows, cols

