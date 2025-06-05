import pandas as pd
import tempfile
from typing import Union, Dict, List
import os, glob
import json
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import gaussian_kde
import calendar
import psutil
import signal
import subprocess
import time
import io
import math
import duckdb


if "refresh_flag" not in st.session_state:
    st.session_state["refresh_flag"] = False


def refresh_visualizer():
    # Toggle the refresh flag to trigger reprocessing
    st.session_state["refresh_flag"] = not st.session_state["refresh_flag"]



def replace_content(data):
    if isinstance(data, dict):
        return {
            replace_content(key): replace_content(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [replace_content(item) for item in data]
    elif isinstance(data, str):
        return data.replace("year", "time").replace("DOY", "dayofyr")
    else:
        return data
    
def dataframe(data, key_name = "summary"):
    header = data['outputs'][key_name]['header']
    data = data['outputs'][key_name]['data']

    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data, columns=header)
    if 'dayofyr' in df.columns:
        df['dayofyr'] = df['dayofyr'].astype(int)
    
    return df

def format_headers(df):
    # trim white spaces from df.columns
    df.columns = [col.strip() for col in df.columns]
    
def convert_to_end_of_month(fractional_time):
    # Extract the year and month from the fractional time
    year = int(fractional_time)
    month_decimal = fractional_time - year
    month = round((month_decimal * 12) % 12) + 1
    
    # Add 1 to the month to get to the next month, then subtract 1 day to get the last day of the current month
    next_month = month + 1 if month < 12 else 1  # Ensure month stays within 1-12 range
    
    try:
        if next_month != 1:
            end_of_month = pd.to_datetime(f"{year}-{next_month}-01") - pd.Timedelta(days=1)
        else:
            end_of_month = pd.to_datetime(f"{year+1}-01-01") - pd.Timedelta(days=1)
        return end_of_month.date()
    except Exception as e:
        return ""

    
def add_date_column(df):
    """
    Add a 'date' column to a DataFrame based on the 'time' and 'dayofyr' columns.

    Args:
        df (pandas.DataFrame): The DataFrame to add the 'date' column to.

    Returns:
        pandas.DataFrame: The DataFrame with the added 'date' column.
    """

    # Check if the 'time' and 'dayofyr' columns exist in the DataFrame.
    if 'time' not in df.columns or 'dayofyr' not in df.columns:
        # raise ValueError("'time' and 'dayofyr' columns must exist in the DataFrame.")
        print("Error: 'time' and 'dayofyr' columns must exist in the DataFrame.")

    # Check if the 'time' column contains non-null values.
    if df['time'].isnull().any():
        # raise ValueError("'time' column contains null values.")
        print("Error: 'time' column contains null values.")

    # Check if the 'dayofyr' column contains non-null values.
    if df['dayofyr'].isnull().any():
        # raise ValueError("'dayofyr' column contains null values.")
        print("Error: 'dayofyr' column contains null values.")
        
    # convert time column to float with rounding to 2
    df['time']= df['time'].astype(float).round(2)
    
    # drop rows with negative time values
    df = df[df['time'] > 100]  
    
    # Convert 'time' and 'dayofyr' columns to strings and concatenate them.
    df['date'] = pd.to_datetime(
        df['time'].astype(int).astype(str) + df['dayofyr'].astype(str),
        format='%Y%j'
    )
    return df
    
def filter_by_time_range(df, min_date, max_date):
    """
    Filter a DataFrame by a time range.

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        min_date (datetime.datetime or None): The minimum date for filtering.
            If None, it is set to the minimum date in the DataFrame.
        max_date (datetime.datetime or None): The maximum date for filtering.
            If None, it is set to the maximum date in the DataFrame.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    
    # Make a copy of the input DataFrame
    df_ = df.copy()
    
    # Get the global minimum and maximum dates from the DataFrame
    min_date_global = df_['date'].min()
    max_date_global = df_['date'].max()
    
    # If min_date is not provided, set it to the global minimum
    if not min_date:
        min_date = min_date_global
        
    # If max_date is not provided, set it to the global maximum
    if not max_date:
        max_date = max_date_global
        
    # Convert min_date and max_date to Timestamp
    min_date = pd.Timestamp(min_date)  
    max_date = pd.Timestamp(max_date)  
    
    # Filter the DataFrame based on the time range
    df_ = df_[(df_['date'] >= min_date) & (df_['date'] <= max_date)]
    
    return df_

def clean_up_df(df, params=None, start_date=None, end_date=None, add_id=None):
    if params:
        df = df[params]
    
    if start_date or end_date:
        df = filter_by_time_range(df, start_date, end_date)
        
    if add_id:
        df['ID'] = add_id
    
    # drop duplicated columns
    df = df.loc[:,~df.columns.duplicated()]
    
    # drop duplicated rows keep last
    df = df.drop_duplicates(subset=df.columns.tolist(), keep='last')
    
    # drop empty columns
    df = df.loc[:,df.apply(pd.Series.nunique)>0]
    
    # drop columns that contain non-numeric values
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1)
    
    # drop columns that are all 0
    df = df.loc[:,df.apply(lambda x: x.eq(0).all()==False)]
    
    # drop columns that are all -99.900002
    df = df.loc[:,df.apply(lambda x: x.eq(-99.900002).all()==False)]

    return df

def return_daily_outputs(outputs: Union[str, List[pd.DataFrame], pd.DataFrame],
                   params: List[str] = None,
                   start_date: str = None,
                   end_date: str = None,
                   add_id: str = None,
                   debug: bool = True):
    '''
    Open output json and return a dataframe of the selected parameters, filtered by start_date and end_date
    search_params allows relative matching of the params with the output parameters from the model
    if regex=True, params can be a list of generic search phrases (e.g., ['aet', 'CH4', 'grain'])'''
    
    df_list = []
    # if outputs is a directory
    if isinstance(outputs, str) and os.path.isdir(outputs):
        # read all the csv files in the directory
        csv_files = glob.glob(os.path.join(outputs, "*.csv"))
        # Exclude csv files containing ['tgmonth.csv', '*_lis.csv', 'year_cflows.csv', 'year_summary.csv']
        exclude_list = ['tgmonth.csv', '_lis', 'year_cflows.csv',
                        'year_summary.csv', 'stemp_dx.csv'] #'harvest.csv', 'harvestgt.csv'
        csv_files  = [f for f in csv_files if not any(x in f for x in exclude_list)]
        
        # read data into dataframes and format header to remove trailing spaces
        for csv_file in csv_files:
            print(csv_file)
            df_ = pd.read_csv(csv_file)
            format_headers(df_)
            if 'time' in df_.columns and 'dayofyr' in df_.columns:
                df_ = add_date_column(df_)
                df_.set_index('date', inplace=True)
                df_ = df_[~df_.index.duplicated(keep='first')]
                # print(df_.columns)
                df_list.append(df_)
        if debug:
            print(f"Found {len(df_list)} daily csv files in the directory: {outputs}")
        
    elif isinstance(outputs, list):
        df_list = outputs
        
        # format headers to remove trailing spaces
        for df in df_list:
            format_headers(df)
        
        # filter daily dataframes only: 
        df_list = [df_ for df_ in df_list if 'time' in df_.columns and 'dayofyr' in df_.columns ]
        if debug:
            print(f"Found {len(df_list)} daily dataframes in the list")
        
        # add the date column
        df_list = [add_date_column(df_) for df_ in df_list]
        for i in range(len(df_list)):
            df_list[i].set_index('date', inplace=True)
            df_list[i] = df_list[i][~df_list[i].index.duplicated(keep='first')]
        
    elif isinstance(outputs, pd.DataFrame):
        df_list = [outputs]
    else:
        raise ValueError(f"Unexpected type: {type(outputs)}")
        
    # merge all the dataframes into a single dataframe based on the date column
    if len(df_list) > 0:
        df = pd.concat(df_list, axis=1, join='outer',)
    else:
        raise ValueError("No dataframes to merge")
    
    
    #df = clean_up_df(df, params, start_date, end_date, add_id)
    
    if debug:
        print(f"Output dataframe shape: {df.shape}")
        #print(df)
        
    return df


def return_harvest_outputs(outputs: Union[str, pd.DataFrame],
                   params: List[str] = None,
                   start_date: str = None,
                   end_date: str = None,
                   add_id: str = None,
                   debug: bool = True,
                   harvest_name: str = 'harvest.csv'):
    
    if isinstance(outputs, str) and os.path.isdir(outputs):
        csv_files = glob.glob(os.path.join(outputs, "*.csv"))
        csv_files = [f for f in csv_files if harvest_name in f]
        if len(csv_files) > 0:
            print(csv_files[0])
            df = pd.read_csv(csv_files[0])
        else:
            raise ValueError(f"No {harvest_name} file found in the directory")
        
    else:
        df = outputs
    
    if not df.empty:
        format_headers(df)
        df = add_date_column(df)
        df.set_index('date', inplace=True)
        df = clean_up_df(df, params, start_date, end_date, add_id)
        
    if debug:
        print(f"Output dataframe shape: {df.shape}")
    return df


def process_json_to_dataframes(json_file):
    # Load and process the JSON file
    with open(json_file, 'r') as file:
        raw_data = json.load(file)
        
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)

    data = replace_content(raw_data)
    
    keys = list(data['outputs'].keys())
    dataframes = [dataframe(data, key) for key in keys if key not in ['harvest', 'aggregated_outputs']]
    
    # Combine dataframes based on daily outputs
    df_combined = return_daily_outputs(dataframes, debug=False)
    
    
    # Add a year column based on the index (assumed to be a datetime object)
    df_combined['year'] = df_combined.index.year

    # Exclude columns that are not relevant for yearly sums
    exclude_columns = ['time', 'dayofyr', 'tmax', 'tmin']
    columns_to_sum = [col for col in df_combined.columns if col not in exclude_columns and col != 'year']

    # Group by year and calculate sum for relevant columns
    df_yearly_sum = df_combined.groupby('year')[columns_to_sum].sum()

    # If 'harvest' data exists, process harvest-related outputs
    if 'harvest' in keys:
        harvest_index = keys.index('harvest')
        df_harvest = return_harvest_outputs(dataframe(data, keys[harvest_index]), debug=False)
        df_harvest.index = pd.to_datetime(df_harvest.index)

        # Add a year column to harvest data
        df_harvest['year'] = df_harvest.index.year
        
        # Exclude columns not relevant for summation
        exclude_columns_harvest = ['time', 'dayofyr']
        columns_to_sum_harvest = [col for col in df_harvest.columns if col not in exclude_columns_harvest and col != 'year']

        # Group by year and calculate sum for harvest data
        df_yearly_sum_harvest = df_harvest.groupby('year')[columns_to_sum_harvest].sum()
    else:
        df_yearly_sum_harvest = None

    if 'aggregated_outputs' in keys:
        df_ao = dataframe(data, 'aggregated_outputs')
        df_ao['year'] = np.floor(df_ao['time']).astype(int)

        # Define column groups
        time_max_columns = ['somsc', 'volpac', 'strmac(2)']
        value_max_columns = ['cinput', 'cproda']

        # Filter available columns
        available_time_max = [col for col in time_max_columns if col in df_ao.columns]

        result_dfs = []

        # Process time_max columns
        if available_time_max:
            idx_time_max = df_ao.groupby('year')['time'].idxmax()
            df_time_max = df_ao.loc[idx_time_max, ['year'] + available_time_max]
            df_time_max.set_index('year', inplace=True)
            result_dfs.append(df_time_max)

        # Process value_max columns (even if not available, fill with None)
        if value_max_columns:
            value_max_results = []

            for col in value_max_columns:
                if col in df_ao.columns:
                    idx_max = df_ao.groupby('year')[col].idxmax()
                    df_col_max = df_ao.loc[idx_max, ['year', col]]
                    df_col_max.set_index('year', inplace=True)
                else:
                    unique_years = df_ao['year'].unique()
                    df_col_max = pd.DataFrame({col: [None]*len(unique_years)}, index=unique_years)
                    df_col_max.index.name = 'year'

                value_max_results.append(df_col_max)

            df_value_max = pd.concat(value_max_results, axis=1, join='outer')
            result_dfs.append(df_value_max)

        if result_dfs:
            df_ao_summary = pd.concat(result_dfs, axis=1, join='outer')
            df_ao_summary = df_ao_summary.sort_index()

    # Clean duplicate columns
    df_yearly_sum = df_yearly_sum.loc[:, ~df_yearly_sum.columns.duplicated(keep='last')]

    combined_df_list = [df for df in [df_yearly_sum, df_ao_summary] if df is not None]
    concat_df = pd.concat(combined_df_list, axis=1, join="outer")
    concat_df = concat_df.loc[:, ~concat_df.columns.duplicated(keep='last')]

    concat_df = concat_df.sort_index()

    return concat_df

class GHGRequiredVars:
    """Class to store and check required variables for GHG calculations."""
    
    REQUIRED_VARS = {
        'somsc': 'aggregated_outputs',
        'N2Oflux': 'summary',
        'NOflux': 'summary',
        'volpac': 'aggregated_outputs',
        'strmac(2)': 'aggregated_outputs',
        'CH4_Ebl': 'methane',
        'CH4_Ep': 'methane'
    }
    
    OPTIONAL_VARS = ['CH4_oxid', 'CH4oxid']  # At least one is required
    
    @classmethod
    def check_missing_vars(cls, df):
        """
        Check for missing required variables in the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            list: Missing variable names with sources
        """
        missing_vars = [
            f"{var} from {source}" 
            for var, source in cls.REQUIRED_VARS.items() 
            if var not in df.columns
        ]

        # Check for at least one CH4 oxidation variable
        if not any(var in df.columns for var in cls.OPTIONAL_VARS):
            missing_vars.append("CH4_oxid/CH4oxid from methane/summary")
        
        return missing_vars

class Constants:
    """A class to store global warming potentials, molecular weights, and emission factors."""
    
    # Global Warming Potentials (100-year time horizon, AR6 values)
    GWP_N2O = 273   # Global warming potential of nitrous oxide (N2O) in g CO2e / g N2O
    GWP_CH4 = 27    # Global warming potential of methane (CH4) in g CO2e / g CH4
    
    # IPCC 2019 Indirect Emission Factors
    EF_LR = 0.011  # Emission factor for leaching/runoff (g N2O-N / g N leached)
    EF_V = 0.01     # Emission factor for volatilized nitrogen (g N2O-N / g N volatilized)
    
    # Emission Factors
    EF_CH4_BIOMASS_BURNING = 2.7    # kg CH4/Mg dry matter
    EF_N2O_BIOMASS_BURNING = 0.07   # kg N2O/Mg dry matter
    EF_DIESEL = 0.002886            # Mg CO2e/L from diesel combustion
    
    # Gram molecular weights (gmw)
    GMW_C = 12      # Gram molecular weight of carbon (C) in g C / mole
    GMW_N = 14      # Gram molecular weight of nitrogen (N) in g N / mole
    GMW_CH4 = 16    # Gram molecular weight of methane (CH4) in g CH4 / mole
    GMW_CO2 = 44    # Gram molecular weight of carbon dioxide (CO2) in g CO2 / mole
    GMW_N2O = 44    # Gram molecular weight of nitrous oxide (N2O) in g N2O / mole
    
    # Conversion Factors
    N_TO_N2O = GMW_N2O/(2*GMW_N)     # 44/28: Conversion from N to N2O2*GMW_N     # 44/28: Conversion from N to N2O
    C_TO_CO2 = GMW_CO2/GMW_C     # 44:12 Conversion from C to CO2
    C_TO_CH4 = GMW_CH4/GMW_C     # 16/12: Conversion from C to CH4
    
    # Unit conversion factors
    KG_TO_MG = 0.001     # Conversion from Kilogram to Megagram
    G_TO_MG = 0.000001   # Conversion from Gram to Megagram
    G_M2_TO_MG_HA = 0.01 # Conversion from g/m² to Mg/ha
    
    # Biomass conversion factors
    COMBUSTION_FACTOR_RICE = 0.8   # Fraction of dry matter burnt for rice
    DM_TO_C_CROP = 0.45           # Conversion from Dry Matter to Carbon for rice straw
    C_TO_DM_CROP = 1/DM_TO_C_CROP    # 1/DM_TO_C_STRAW: Conversion from Carbon to Dry Matter for rice straw 
    DM_TO_C_WOOD = 0.5           # Conversion from Dry Matter to Carbon for wood
    C_TO_DM_WOOD = 1/DM_TO_C_WOOD    # 1/DM_TO_C_WOOD: Conversion from Carbon to Dry Matter for wood

    # Example crop mapping dictionary
    factor_mapping = {
        # Crop types
        "ct": "Cotton",
        "ct-m": "Cotton Monocrop",
        "gn": "Groundnut",
        "gn-m": "Groundnut Monocrop",
        "wt": "Wheat",

        # Residue management
        "noRI": "No Residue Incorporation",
        "RI": "Residue Incorporation",

        # Irrigation methods
        "furr": "Furrow Irrigation",
        "drip": "Drip Irrigation",

        # Cover cropping
        "nocc": "No Cover Cropping",
        "cc": "Cover Cropping",

        "wR0": "wR0",
        "wR90": "wR90",
        "wR20": "wR20"
    }



def calc_GHG(concatlis, scenario=False):
    """ 
    Calculate greenhouse gas (GHG) emissions, carbon flux, and related metrics.
    
    Parameters: 
        concatlis (pd.DataFrame): Input DataFrame containing raw data.
        scenario (bool): Whether to include scenario column in output.
    
    Returns: 
        tuple: (DataFrame, list of missing variables or error message) 
    """
    try:
        # Check if 'Scenario' column exists and has multiple unique values
        if 'Scenario' in concatlis.columns and len(concatlis['Scenario'].unique()) >= 2:
            # Initialize an empty DataFrame to store combined results
            combined_results = None
            
            # Process each scenario separately
            for scenario_value in concatlis['Scenario'].unique():
                # Filter data for this scenario
                scenario_data = concatlis[concatlis['Scenario'] == scenario_value]
                
                # Check for required variables in this scenario subset
                missing_vars = GHGRequiredVars.check_missing_vars(scenario_data)
                if missing_vars:
                    return None, f"Missing variables for scenario {scenario_value}: {missing_vars}"
                
                # Calculate GHG for this scenario
                scenario_ghg = calculate_single_scenario_ghg(scenario_data)
                
                # Add scenario identifier
                scenario_ghg['Scenario'] = scenario_value
                
                # Merge with combined results
                if combined_results is None:
                    combined_results = scenario_ghg
                else:
                    combined_results = pd.concat([combined_results, scenario_ghg])
            
            return combined_results, None
        else:
            # Original single-scenario processing
            missing_vars = GHGRequiredVars.check_missing_vars(concatlis)
            if missing_vars:
                return None, missing_vars
            
            result_df = calculate_single_scenario_ghg(concatlis)
            
            # Add Scenario column if requested
            if scenario and 'Scenario' in concatlis.columns:
                result_df['Scenario'] = concatlis['Scenario']
            
            return result_df, None
            
    except Exception as e:
        return None, str(e)

def calculate_single_scenario_ghg(data):
    """Helper function to calculate GHG emissions for a single dataset"""
    # CO2 flux calculation (convert from C to CO2) 
    CO2FLUX = -(data['somsc'] - data['somsc'].shift()) * Constants.G_M2_TO_MG_HA  # Convert g/m2 to Mg/ha 
    CO2FLUX *= Constants.C_TO_CO2  # Convert from C to CO2 
     
    # Direct N2O emissions (converted to CO2-equivalent) default(g N ha−1 d−1) 
    direct_N2O = data['N2Oflux'] * Constants.G_TO_MG * Constants.GWP_N2O * Constants.N_TO_N2O  
     
    # CH4 flux calculations (handle different variable names) 
    if 'CH4_oxid' in data.columns: 
        CH4OXID = data['CH4_oxid'] * Constants.G_M2_TO_MG_HA  # Convert g/m² to Mg/ha 
    elif 'CH4oxid' in data.columns: 
        CH4OXID = data['CH4oxid'] * Constants.G_TO_MG  # Convert g/ha to Mg/ha 
    else: 
        CH4OXID = 0  # Default if column is missing 

    CH4_emissions = (data['CH4_Ep'] + data['CH4_Ebl']) * Constants.G_M2_TO_MG_HA  # Convert g/m² to Mg/ha 
    CH4_total = (-CH4OXID + CH4_emissions) * Constants.GWP_CH4 * Constants.C_TO_CH4 

    # Indirect N2O emissions 
    indirect_N2O_vol = ( 
        Constants.EF_V *  
        (data['NOflux'] * Constants.G_TO_MG + data['volpac'] * Constants.G_M2_TO_MG_HA) * 
        Constants.GWP_N2O *  
        Constants.N_TO_N2O 
    ) 
     
    indirect_N2O_NO3 = ( 
        data['strmac(2)'] * Constants.G_M2_TO_MG_HA *  
        Constants.EF_LR *  
        Constants.GWP_N2O *  
        Constants.N_TO_N2O 
    ) 

    # Total farming GHG emissions 
    Farming_GHG = direct_N2O + indirect_N2O_NO3 + indirect_N2O_vol + CH4_total + CO2FLUX 

    # Construct output DataFrame 
    ghg_df = data[['somsc']].copy()  # Start with one column to initialize 
    ghg_df['CO2 (MgCO2e/ha)'] = CO2FLUX 
    ghg_df['CH4 (MgCO2e/ha)'] = CH4_total 
    ghg_df['Direct_N2O (MgCO2e/ha)'] = direct_N2O 
    ghg_df['Indirect_N2O_vol (MgCO2e/ha)'] = indirect_N2O_vol 
    ghg_df['Indirect_N2O_leached (MgCO2e/ha)'] = indirect_N2O_NO3 
    ghg_df['Total_Soil_GHG (MgCO2e/ha)'] = Farming_GHG
    ghg_df['cinput'] = data[['cinput']]
    ghg_df['cproda'] = data[['cproda']]
    
    return ghg_df
    
def calculate_mean_ghg_by_scenario(df, start_year=None, end_year=None):
        """
        Calculate the mean GHG values for each scenario within a given year range.
        
        If start_year and/or end_year are None, the filter will use the entire date range.
        
        Parameters:
        - df: DataFrame containing GHG data.
        - start_year: (Optional) Start year for filtering the data.
        - end_year: (Optional) End year for filtering the data.
        
        Returns:
        - DataFrame with mean GHG values grouped by Scenario.
        """
        # Create a filtered dataframe based on the presence of start_year and end_year.
        filtered_df = df.copy()
        if start_year is not None:
            filtered_df = filtered_df[filtered_df.index >= start_year]
        if end_year is not None:
            filtered_df = filtered_df[filtered_df.index <= end_year]
        
        mean_values = filtered_df.groupby("Scenario").mean(numeric_only=True).reset_index(drop=True)

        return mean_values



def process_json_with_result(file_input):
    try:
        if isinstance(file_input, str) and os.path.isfile(file_input):
            df = process_json_to_dataframes(file_input)  # Only call process_json_to_dataframes if file_input is a file path
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                temp_file.write(file_input.read())  # Create a temporary file from file_input
                temp_file_path = temp_file.name

            # Process the temporary file
            df = process_json_to_dataframes(temp_file_path)  # Call process_json_to_dataframes for the temporary file
            os.remove(temp_file_path)  # Delete the temporary file after processing
    except Exception as e:
        df = None
    return df

def process_json_without_result(file_input):
    try:
        # Check if file_input is a valid file path (string)
        if isinstance(file_input, str) and os.path.isfile(file_input):
            # If file_input is a valid file path, directly process it
            df1, df2, df3 = non_result(file_input)
        
        # Check if file_input is a file-like object (e.g., uploaded file from Streamlit)
        else:
            # If file_input is a file-like object, write the data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                temp_file_path = temp_file.name  # Get the temporary file's path
                temp_file.write(file_input.read())  # Write the uploaded file data to the temp file

            # Process the temporary file
            df1, df2, df3 = non_result(temp_file_path)
            os.remove(temp_file_path)  # Remove the temporary file after processing

    except Exception as e:
        # If an error occurs, log it and return None values for the DataFrames
        df1, df2, df3 = None, None, None

    # Return the processed DataFrames (or None if an error occurred)
    return df1, df2, df3

def remove_high_nan_columns(daily_data, df_harvest, df_ao_base):
    """
    Process DataFrames and remove columns that have 50% or more NaN values.
    Returns the processed DataFrames.
    """
    def process_single_df(df):
        if df is None:
            return None
            
        # Calculate the percentage of NaN values in each column
        nan_percentage = df.isna().mean()
        
        # Get columns where NaN percentage is less than 50%
        columns_to_keep = nan_percentage[nan_percentage < 0.5].index
        
        # Return DataFrame with only the kept columns
        return df[columns_to_keep]
    
    # Process daily data (tuple of list_daily_dfs and daily_keys)
    list_daily_dfs, daily_keys = daily_data
    processed_daily_dfs = []
    
    for df in list_daily_dfs:
        processed_df = process_single_df(df)
        processed_daily_dfs.append(processed_df)
    
    processed_daily_data = (processed_daily_dfs, daily_keys)
    
    # Process other DataFrames
    processed_harvest = process_single_df(df_harvest)
    processed_ao_base = process_single_df(df_ao_base)
    
    return processed_daily_data, processed_harvest, processed_ao_base


def remove_invalid_columns(df_combine, df_harvest, df_ao_base):
    """
    Process DataFrames and remove:
    1. Columns that have 50% or more NaN values
    2. Columns that are not numeric (can't be used in time series)
    Returns the processed DataFrames.
    """
    def process_single_df(df):
        if df is None:
            return None
            
        # Calculate the percentage of NaN values in each column
        nan_percentage = df.isna().mean()
        
        # Get columns where NaN percentage is less than 50%
        columns_to_keep = nan_percentage[nan_percentage < 0.5].index
        df = df[columns_to_keep]
        
        # Keep only numeric columns (int, float)
        numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        df = df[numeric_columns]
        
        return df
    
    # Process each DataFrame
    processed_combine = process_single_df(df_combine)
    processed_harvest = process_single_df(df_harvest)
    processed_ao_base = process_single_df(df_ao_base)
    
    return processed_combine, processed_harvest, processed_ao_base

    
def non_result(json_file):
    try:
        with open(json_file, 'r') as file:
            raw_data = json.load(file)
    except:
        return None, None, None

    # load json
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except:
            return None, None, None

    # process
    try:
        data = replace_content(raw_data)
        if not isinstance(data, dict):
            return None, None, None

        if 'outputs' not in data or not isinstance(data['outputs'], dict):
            return None, None, None

        keys = list(data['outputs'].keys())
        daily_keys = [key for key in keys if key not in ['harvest', 'aggregated_outputs']]
        dataframes = [dataframe(data, key) for key in daily_keys]

        if dataframes:
            # Calculate somsc for each dataframe individually
            list_daily_dfs = []
            for df in dataframes:
                df_with_somsc = calc_somsc(df)
                list_daily_dfs.append(df_with_somsc)
            
            daily_data = (list_daily_dfs, daily_keys)
        else:
            daily_data = ([], [])

        # Check for 'harvest' and 'aggregated_outputs'
        df_harvest = dataframe(data, 'harvest') if 'harvest' in keys else None
        df_ao_base = dataframe(data, 'aggregated_outputs') if 'aggregated_outputs' in keys else None

        return daily_data, df_harvest, df_ao_base

    except Exception as e:
        return None, None, None


def merge_dataframes_daily_vars(dataframes):
    """
    Merge a list of DataFrames horizontally based on 'time' and 'dayofyr' columns,
    removing duplicate columns during the merge process.
    
    Parameters:
    - dataframes: list of pandas DataFrames to merge.
    
    Returns:
    - A merged DataFrame if valid DataFrames are found, else None.
    """
    if not dataframes:
        return None
        
    # Get the first DataFrame as base
    merged_df = dataframes[0]
    
    # Merge each subsequent DataFrame
    for df in dataframes[1:]:
        # Create a list of columns that would be duplicated (excluding merge keys)
        duplicate_cols = [col for col in df.columns 
                         if col in merged_df.columns and col not in ['time', 'dayofyr']]
        
        # Drop duplicate columns from the right DataFrame before merging
        df_clean = df.drop(columns=duplicate_cols)
        
        # Merge DataFrames
        merged_df = pd.merge(merged_df, df_clean, 
                           on=['time', 'dayofyr'], 
                           how='outer')
    
    return merged_df


def calculate_time_series_metrics(series, selected_metrics):
    """Calculate time series statistics based on selected metrics"""
    stats_dict = {}
    
    if 'basic' in selected_metrics:
        stats_dict.update({
            'Maximum': series.max(),
            'Minimum': series.min(),
            'Mean': series.mean(),
            'Median': series.median(),
            'Std Dev': series.std()
        })
    
    if 'distribution' in selected_metrics:
        stats_dict.update({
            'Skewness': series.skew(),
            'Kurtosis': series.kurtosis(),
            'Q1': series.quantile(0.25),
            'Q3': series.quantile(0.75),
            'IQR': series.quantile(0.75) - series.quantile(0.25)
        })
    
    if 'time_series' in selected_metrics:
        # Rolling statistics
        stats_dict.update({
            'Rolling Mean (7-period)': series.rolling(window=7).mean().iloc[-1],
            'Rolling Std (7-period)': series.rolling(window=7).std().iloc[-1]
        })
        
        # Growth rates
        pct_change = series.pct_change().dropna()
        stats_dict.update({
            'Avg Growth Rate (%)': pct_change.mean() * 100,
            'Growth Volatility (%)': pct_change.std() * 100
        })
        
        try:
            decomposition = seasonal_decompose(series, period=12, extrapolate_trend='freq')
            stats_dict.update({
                'Trend (Last Value)': decomposition.trend.iloc[-1],
                'Seasonal (Last Value)': decomposition.seasonal.iloc[-1]
            })
        except:
            pass
        
        try:
            adf_result = adfuller(series.dropna())
            stats_dict.update({
                'ADF Statistic': adf_result[0],
                'ADF p-value': adf_result[1]
            })
        except:
            pass
    
    if 'autocorrelation' in selected_metrics:
        try:
            acf_result = acf(series.dropna(), nlags=1)[1]
            pacf_result = pacf(series.dropna(), nlags=1)[1]
            stats_dict.update({
                'Lag-1 Autocorrelation': acf_result,
                'Partial Autocorrelation': pacf_result
            })
        except:
            pass
    
    return stats_dict


def show_calculated_components(df, show_stats=True, plot_height=600):
    """
    Generate density plots for GHG components over time, grouped by scenarios.
    Each scenario is processed in a try/except block so that an error in one
    doesn't affect others.
    """

    ghg_components = [
        'CO2 (MgCO2e/ha)', 
        'CH4 (MgCO2e/ha)', 
        'Direct_N2O (MgCO2e/ha)', 
        'Indirect_N2O_vol (MgCO2e/ha)', 
        'Indirect_N2O_leached (MgCO2e/ha)'
    ]

    ghg_components = ['CO2 (MgCO2e/ha)', 'CH4 (MgCO2e/ha)', 
                     'Direct_N2O (MgCO2e/ha)', 'Indirect_N2O_vol (MgCO2e/ha)', 'Indirect_N2O_leached (MgCO2e/ha)']

    
    # Validate that required columns exist
    for component in ghg_components:
        if component not in df.columns:
            st.warning(f"Column '{component}' does not exist in the DataFrame!")
            return

    # Extract unique scenarios from the DataFrame
    unique_scenarios = df['Scenario'].unique()
    
    for scenario in unique_scenarios:

        try:
            st.header(f"Analysis for Scenario: {scenario}")
            
            # Create the plot for the current scenario
            fig = go.Figure()
            scenario_df = df[df['Scenario'] == scenario]
            
            for component in ghg_components:
                fig.add_trace(
                    go.Scatter(
                        x=scenario_df.index,
                        y=scenario_df[component],
                        mode='lines+markers',
                        name=component,
                        line=dict(width=2)
                    )
                )
            
            fig.update_layout(
                title="Trend of Total Soil GHG Emissions (MgCO2e/ha)",
                xaxis_title="Year",
                yaxis_title="MgCO2e/ha",
                template="plotly_white",
                height=plot_height,
                legend_title="GHG Components",
                hovermode="x unified"
            )
            
            fig.update_traces(
                hovertemplate="<b>Year: %{x}</b><br>" +
                              "Total GHG: %{y}<br>" +
                              "Component: %{fullData.name}<extra></extra>"
            )
            
            # Use a unique key for each plotly_chart
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{scenario}")
            
            if show_stats:
                st.subheader("Statistical Summary", divider='gray')
                
                # Create rows with up to 5 columns per row for the components
                max_cols = 5
                total_components = len(ghg_components)
                num_rows = math.ceil(total_components / max_cols)
                
                for row in range(num_rows):
                    cols = st.columns(max_cols)
                    for col in range(max_cols):
                        idx = row * max_cols + col
                        if idx >= total_components:
                            break
                        component = ghg_components[idx]
                        with cols[col]:
                            st.markdown(f"**{component}**")
                            
                            # Check if there is enough data to calculate statistics
                            if scenario_df[component].empty or len(scenario_df[component]) < 2:
                                st.info("Not enough data to compute statistics.")
                                continue
                            
                            mean = scenario_df[component].mean()
                            median = scenario_df[component].median()
                            std = scenario_df[component].std()
                            
                            first_value = scenario_df[component].iloc[0]
                            last_value = scenario_df[component].iloc[-1]
                            change = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
                            
                            st.markdown(f"""
                            Mean: `{mean:.2f}`  
                            Median: `{median:.2f}`  
                            Std Dev: `{std:.2f}`  
                            Change: `{change:+.1f}%`
                            """)
        except Exception as e:
            st.error(f"Error processing scenario {scenario}: {e}")
            continue


        st.header(f"Analysis for Scenario: {scenario}")
        
        # Create plot
        fig = go.Figure()
        scenario_df = df[df['Scenario'] == scenario]
        
        for component in ghg_components:
            fig.add_trace(
                go.Scatter(
                    x=scenario_df.index,
                    y=scenario_df[component],
                    mode='lines+markers',
                    name=component,
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=f"Trend of Total Soil GHG Emissions (MgCO2e/ha)",
            xaxis_title="Year",
            yaxis_title="MgCO2e/ha",
            template="plotly_white",
            height=plot_height,
            legend_title="GHG Components",
            hovermode="x unified"
        )

        fig.update_traces(
        hovertemplate="<b>Year: %{x}</b><br>" +
                    "Total GHG: %{y}<br>" +
                    "Component: %{fullData.name}<extra></extra>"
    )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if show_stats:
            st.subheader("Statistical Summary", divider='gray')
            cols = st.columns(5)
            
            for idx, component in enumerate(ghg_components):
                with cols[idx]:
                    st.markdown(f"**{component}**")
                    
                    mean = scenario_df[component].mean()
                    median = scenario_df[component].median()
                    std = scenario_df[component].std()
                    
                    first_value = scenario_df[component].iloc[0]
                    last_value = scenario_df[component].iloc[-1]
                    change = ((last_value - first_value) / first_value) * 100
                    
                    st.markdown(f"""
                    Mean: `{mean:.2f}`  
                    Median: `{median:.2f}`  
                    Std Dev: `{std:.2f}`  
                    Change: `{change:+.1f}%`
                    """)

def show_calculated_components_for_all_scenario(df, show_stats=True, plot_height=600):
    """
    Generate separate plots for each GHG component showing all scenarios.
    """
    ghg_components = ['CO2 (MgCO2e/ha)', 'CH4 (MgCO2e/ha)', 
                     'Direct_N2O (MgCO2e/ha)', 'Indirect_N2O_vol (MgCO2e/ha)', 'Indirect_N2O_leached (MgCO2e/ha)']
    
    # Validate columns
    for component in ghg_components:
        if component not in df.columns:
            st.warning(f"Column '{component}' does not exist in the DataFrame!")
            return
    
    # Create a plot for each component
    for component in ghg_components:
        st.subheader(f"{component} Analysis")
        
        fig = go.Figure()
        
        # Add a trace for each scenario
        for scenario in df['Scenario'].unique():
            scenario_df = df[df['Scenario'] == scenario]
            
            fig.add_trace(
                go.Scatter(
                    x=scenario_df.index,
                    y=scenario_df[component],
                    mode='lines+markers',
                    name=scenario,
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=f"{component} Trends Across All Scenarios (MgCO2e/ha)",
            xaxis_title="Year",
            yaxis_title=f"MgCO2e/ha",
            template="plotly_white",
            height=plot_height,
            legend_title="Scenarios",
            hovermode="x unified"
        )
        fig.update_traces(
            hovertemplate="<b>Year: %{x}</b><br>" +
                        "Scenario: %{fullData.name}<br>" +
                        "Value: %{y} MgCO2e/ha<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        if show_stats:
            # Calculate statistics for each scenario
            stats_cols = st.columns(len(df['Scenario'].unique()))
            
            for idx, scenario in enumerate(df['Scenario'].unique()):
                with stats_cols[idx]:
                    scenario_df = df[df['Scenario'] == scenario]
                    st.markdown(f"**{scenario}**")
                    
                    mean = scenario_df[component].mean()
                    median = scenario_df[component].median()
                    std = scenario_df[component].std()
                    
                    first_value = scenario_df[component].iloc[0]
                    last_value = scenario_df[component].iloc[-1]
                    change = ((last_value - first_value) / first_value) * 100
                    
                    st.markdown(f"""
                    Mean: `{mean:.2f}`  
                    Median: `{median:.2f}`  
                    Std Dev: `{std:.2f}`  
                    Change: `{change:+.1f}%`
                    """)


def process_time_column(selected_df):
    # If time column exists, process using existing time column
    if 'time' in selected_df.columns:
        selected_df['days_in_year'] = np.floor(selected_df['time']).astype(int).apply(
            lambda year: 366 if calendar.isleap(year) else 365
        )
        selected_df['time'] = np.floor(selected_df['time']).astype(int) + (selected_df['dayofyr'] - 1) / selected_df['days_in_year']
    
    # If time column doesn't exist but Date column exists
    elif 'Date' in selected_df.columns:
        selected_df['Date'] = pd.to_datetime(selected_df['Date'])
        selected_df['year'] = selected_df['Date'].dt.year
        selected_df['dayofyr'] = selected_df['Date'].dt.dayofyear
        selected_df['days_in_year'] = selected_df['year'].apply(
            lambda year: 366 if calendar.isleap(year) else 365
        )
        selected_df['time'] = selected_df['year'] + (selected_df['dayofyr'] - 1) / selected_df['days_in_year']
    
    return selected_df


def data_transformation_ui(selected_df1, column, obs_data=None):
    if obs_data is None:
        # UI for transforming data when no observed data is provided
        with st.expander(f"🛠️ Transform `{column}`", expanded=False):
            custom_formula = st.text_area(
                "Enter formula (use 'x'):",
                value="",
                height=68,
                key=f"formula_input_{column}",
                help="Enter a Python expression, e.g., 'x * 2' or 'x + 10'"
            )
            
            if custom_formula.strip():  # Check if the formula is not empty
                try:
                    selected_df1[column] = selected_df1[column].apply(eval("lambda x: " + custom_formula))
                    st.success(f"Column `{column}` transformed successfully!")
                except Exception as e:
                    st.error(f"Error in transformation: {e}")
        
        return selected_df1, None, None
    
    else:
        # Split UI into sections when observed data is available
        with st.expander(f"🛠️ Transform and Analyze `{column}`", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Transform section
                st.markdown(f"### 🛠️ Transform `{column}`")
                custom_formula = st.text_area(
                    "Enter formula (use 'x'):",
                    value="",
                    height=68,
                    key=f"formula_input_{column}",
                    help="Enter a Python expression, e.g., 'x * 2' or 'x + 10'"
                )
                
                if custom_formula.strip():  # Apply transformation if formula is not empty
                    try:
                        selected_df1[column] = selected_df1[column].apply(eval("lambda x: " + custom_formula))
                        st.success(f"Column `{column}` transformed successfully!")
                    except Exception as e:
                        st.error(f"Error in transformation: {e}")
            
            with col2:
                # Observed data variables
                st.markdown("### 🧐 Observed Variables")
                obs_columns = [col for col in obs_data.columns 
                               if col not in ['Scenario', 'Date', 'time', 'dayofyr', 'days_in_year', 'year', 'ID']]
                selected_obs_columns = st.multiselect(
                    "Select variables:",
                    obs_columns,
                    key=f"observed_data_multiselect_{column}"
                )
            
            with col3:
                # ID selection
                st.markdown("### 🔢 Select IDs")
                ID_select = st.multiselect(
                    "Choose IDs:", 
                    options=['ID'] + list(obs_data['ID'].unique()),
                    key=f"id_column_selector_{column}"
                )
        
        return selected_df1, selected_obs_columns, ID_select



def plot_scenarios_streamlit(dataframes, obs_data=None):
    """Enhanced function to visualize time series data with Plotly and provide statistical metrics."""
    
    # Combine dataframes and filter by time period
    combined_df = pd.concat(dataframes)
    
    # Sidebar for year range selection
    st.sidebar.markdown("### 📅 Time Period Selection")
    min_year, max_year = int(combined_df.index.min()), int(combined_df.index.max())
    year_range = st.sidebar.slider(
        "Select analysis period:",
        min_year, max_year,
        (min_year, max_year),
        help="Select the years you wish to analyze"
    )
    
    # Filter combined_df by selected year range before merging with obs_data
    combined_df = combined_df[(combined_df.index >= year_range[0]) & (combined_df.index <= year_range[1])]
    
    # Merge with observed data if it exists
    if obs_data is not None:
        obs_data.set_index('time', inplace=True)  # Ensure 'time' column is used as the index
        filtered_df = pd.merge(combined_df, obs_data, how='outer', left_index=True, right_index=True)
    else:
        filtered_df = combined_df  # No observed data, just use the filtered combined_df
    
    # Sidebar for analysis options
    st.sidebar.markdown("### 📊 Analysis Options")
    metrics_options = {
        'basic': 'Basic Statistics',
        'distribution': 'Distribution Statistics',
        'time_series': 'Time Series Metrics',
        'autocorrelation': 'Autocorrelation Analysis'
    }
    
    selected_metrics = st.sidebar.multiselect(
        "Select Statistical Metrics:",
        options=list(metrics_options.keys()),
        default=['basic'],
        format_func=lambda x: metrics_options[x],
        help="Choose which metrics to display"
    )
    
    # Column selection for scenarios
    st.markdown("### 📊 Data Variables (Scenarios)")
    available_columns = [col for col in combined_df.columns if col != 'Scenario']
    selected_columns = st.multiselect(
        "Select variables to analyze (Scenarios):",
        available_columns,
        help="Choose the variables to display in the plots"
    )
    
    if not selected_columns:
        st.info("👆 Please select at least one variable to display the analysis.")
        return
    
    # Create plots and analysis for each selected variable
    for column in selected_columns:
        st.markdown(f"### 📈 Analysis for: {column}")
        
        # Special handling for Farm_GHG calculation display
        if column == 'Total_Soil_GHG (MgCO2e/ha)':
            ghg_components = ['CO2 (MgCO2e/ha)', 'CH4 (MgCO2e/ha)', 
                     'Direct_N2O (MgCO2e/ha)', 'Indirect_N2O_vol (MgCO2e/ha)', 'Indirect_N2O_leached (MgCO2e/ha)', 'Scenario']
        
            # Select only the GHG components and drop rows where all values are NaN
            ghg_df = filtered_df[ghg_components].dropna(how='all')
            
            
            # Create radio buttons for visualization options
            viz_option = st.radio(
                "🌍 **Greenhouse Gas (GHG) Visualization Options**",
                ["None", "🌟 Combined Scenarios View", "🔍 Individual Components View"],
                index=0,  # Set a default selection (e.g., "Combined Scenarios View")
                help="Select how you want to visualize GHG emissions and their components.\n\n"
                     "- **None**: Do not display any visualizations.\n"
                     "- **🌟 Combined Scenarios View**: Show trends for all scenarios together in a single chart.\n"
                     "- **🔍 Individual Components View**: Display trends for each scenario separately, highlighting specific GHG components."
            )

            # Add a subtle horizontal divider for aesthetics
            st.markdown("---")
            
            if viz_option != "None":
                # Add extra options for customization
                with st.expander("Visualization Options"):
                    show_stats = st.checkbox("Show Statistics", value=True)
                    plot_height = st.slider("Plot Height", 400, 800, 600)
                
                if viz_option == "🌟 Combined Scenarios View":
                    show_calculated_components(ghg_df, show_stats, plot_height)
                elif viz_option == "🔍 Individual Components View":
                    show_calculated_components_for_all_scenario(ghg_df, show_stats, plot_height)
        
        selected_df1, selected_obs_columns, ID_select = data_transformation_ui(filtered_df, column, obs_data)
        # Analyze and visualize emissions
        analyze_and_visualize_emissions(selected_df1, column, year_range, selected_metrics, obs_op=selected_obs_columns, ID_select=ID_select)


def analyze_and_visualize_emissions(filtered_df, column, year_range, selected_metrics, obs_op=None, ID_select=None):
    """
    Analyze and visualize greenhouse gas emission trends over time and scenarios.

    Parameters:
        filtered_df (pd.DataFrame): DataFrame filtered by required conditions.
        column (str): The column to plot and analyze.
        year_range (tuple): The time range for analysis (start_year, end_year).
        selected_metrics (list): List of statistics/metrics to calculate.
        obs_op (list, optional): Additional observation columns for scatter plots.
    """
    try:
        # Create figure with multiple trace support
        fig = go.Figure()

        # Add line traces for primary column
        if column:
            for scen in filtered_df['Scenario'].unique():
                scen_df = filtered_df[filtered_df['Scenario'] == scen]
                fig.add_trace(
                    go.Scatter(
                        x=scen_df.index, 
                        y=scen_df[column], 
                        mode='lines+markers',
                        name=f'{scen}',
                        line=dict(width=2)
                    )
                )

        # Add scatter traces for additional columns
        if obs_op:
            for obs in obs_op:
                for scen in ID_select:
                    scen_df = filtered_df[filtered_df['ID'] == scen]
                    fig.add_trace(
                        go.Scatter(
                            x=scen_df.index, 
                            y=scen_df[obs], 
                            mode='markers',
                            name=f'{obs} - {scen}',
                            marker=dict(size=8, opacity=0.7)
                        )
                    )

        # Update layout
        fig.update_layout(
            title=f'Trend of {column} ({year_range[0]} - {year_range[1]}))',
            xaxis_title='Time',
            yaxis_title="MgCO2e/ha",
            template="plotly_white",
            height=600,
            showlegend=True,
            legend_title="Scenario"
        )

        fig.update_traces(
            hovertemplate="<b>Year: %{x}</b><br>" +
                        "Scenario: %{fullData.name}<br>" +
                        "Value: %{y} MgCO2e/ha<extra></extra>"
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Statistical analysis section
        if selected_metrics:
            st.markdown("### 📊 Statistical Analysis")
            unique_scenarios = filtered_df['Scenario'].dropna().unique()
            
            for scen in unique_scenarios:
                st.markdown(f"#### Scenario: {scen}")
                scen_df = filtered_df[filtered_df['Scenario'] == scen]
                
                # Analyze primary column
                stats_results = calculate_time_series_metrics(scen_df[column], selected_metrics)
                
                # Display metrics in columns
                metrics_per_row = 5
                metrics_items = list(stats_results.items())

                for i in range(0, len(metrics_items), metrics_per_row):
                    cols = st.columns(metrics_per_row)
                    for j in range(metrics_per_row):
                        if i + j < len(metrics_items):
                            metric, value = metrics_items[i + j]
                            with cols[j]:
                                try:
                                    formatted_value = f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
                                    st.markdown(
                                        f"<div style='font-size: 12px;'><b>{metric}:</b> {formatted_value}</div>",
                                        unsafe_allow_html=True
                                    )
                                except Exception as e:
                                    st.markdown(f"<div style='font-size: 12px;'>Error: {str(e)}</div>", unsafe_allow_html=True)

        # Time series decomposition section
        if "time_series" in selected_metrics:
            st.sidebar.markdown("### 🌦️ Seasonal Period Settings")
            
            for scen in unique_scenarios:
                seasonal_period_key = f"seasonal_period_slider_{scen}_{column}"
                seasonal_period = st.sidebar.slider(
                    "Select Seasonal Period (e.g., 12 for monthly, 4 for quarterly):",
                    min_value=1, max_value=24, value=12,
                    key=seasonal_period_key
                )

                st.markdown(f"### 📈 Time Series Components for Scenario: {scen}")
                scen_df = filtered_df[filtered_df['Scenario'] == scen]

                # Handle missing values
                time_series = scen_df[column].dropna()

                if time_series.empty:
                    st.warning(f"No data available for Scenario: {scen} after removing missing values.")
                    continue

                try:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(
                        time_series,
                        period=seasonal_period,
                        extrapolate_trend='freq'
                    )

                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Trend', 'Seasonal', 'Residual'),
                        vertical_spacing=0.1
                    )

                    # Add traces
                    fig.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'),
                        row=3, col=1
                    )

                    fig.update_layout(height=800, title_text=f"Time Series Decomposition - Scenario: {scen}")
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not perform seasonal decomposition for Scenario: {scen}. Error: {e}")

            st.markdown("---")  # Separator between scenarios

    except Exception as e:
        pass

def calc_somsc(df):
    columns = ["som1c(2)", "som2c(2)", "som3c"]
    
    # Check if all columns exist in the DataFrame
    if all(col in df.columns for col in columns):
        df["somsc"] = df[columns].sum(axis=1)
    else:
        return df  # Return the original DataFrame if any column is missing
    
    return df  # Return the updated DataFrame with the new 'somsc' column


def process_csv_without_result(folder_path):
    """
    Processes CSV files in the specified folder. Looks for files containing '_aggregated_outputs.csv' 
    and '_daily_outputs.csv' in their names. If multiple files are found, they are merged 
    into a single DataFrame. Adds a 'Scenario' column extracted from the file name 
    and splits it based on the first character.

    Args:
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        tuple: A tuple containing DataFrames for '_aggregated_outputs.csv' and '_daily_outputs.csv'. 
               If no matching files are found, the corresponding DataFrame will be None.
    """
    # Initialize lists for storing file paths
    wlis_files = []
    daily_vars_files = []

    # Iterate through the folder to find the target files: _daily_outputs.csv, _aggregated_outputs.csv
    for file_name in os.listdir(folder_path):
        if '_aggregated_outputs.csv' in file_name:
            wlis_files.append(os.path.join(folder_path, file_name))
        elif '_daily_outputs.csv' in file_name:
            daily_vars_files.append(os.path.join(folder_path, file_name))

    # Read and merge 'wlis.csv' files
    try:
        if wlis_files:
            wlis_dfs = []
            for wlis_file in wlis_files:
                # Read the file
                temp_df = pd.read_csv(wlis_file)
                # Add the 'Scenario' column based on the file name
                scenario = os.path.basename(wlis_file).split('/')
                temp_df['Scenario'] = scenario[-1].replace('_aggregated_outputs.csv', '')
                wlis_dfs.append(temp_df)
            wlis_df = pd.concat(wlis_dfs, axis=0, ignore_index=True, join='outer')
        else:
            wlis_df = None  # Handle case when no files are provided
    except:
        wlis_df = None

    # Read and merge 'daily_vars.csv' files
    try:
        if daily_vars_files:
            daily_vars_dfs = []
            for daily_vars_file in daily_vars_files:
                # Read the file
                temp_df = pd.read_csv(daily_vars_file)
                # Add the 'Scenario' column based on the file name
                scenario = os.path.basename(daily_vars_file).split('/')
                temp_df['Scenario'] = scenario[-1].replace('_daily_outputs.csv', '')
                daily_vars_dfs.append(temp_df)
            daily_vars_df = pd.concat(daily_vars_dfs, axis=0, ignore_index=True, join='outer')
            daily_vars_df = calc_somsc(daily_vars_df)
        else:
            daily_vars_df = None
    except:
        daily_vars_df = None

    return wlis_df, daily_vars_df

def process_csv_with_result(folder_path):
    """
    Processes CSV files in the specified folder, computes yearly summaries for each scenario,
    and combines results vertically if multiple scenarios exist.
    
    Args:
        folder_path (str): Path to the folder containing the CSV files.
    
    Returns:
        pd.DataFrame: Combined DataFrame containing yearly summaries from all scenarios.
        None: If any error occurs during processing.
    """
    try:
        # Call the original function to get wlis_df and daily_vars_df
        wlis_df, daily_vars_df = process_csv_without_result(folder_path)
        
        if wlis_df is None and daily_vars_df is None:
            return None
        
        # Initialize list to store processed dataframes for each scenario
        processed_dfs = []
        
        # Get unique scenarios
        scenarios = []
        if wlis_df is not None and 'Scenario' in wlis_df.columns:
            scenarios.extend(wlis_df['Scenario'].unique())
        if daily_vars_df is not None and 'Scenario' in daily_vars_df.columns:
            scenarios.extend(daily_vars_df['Scenario'].unique())
        scenarios = list(set(scenarios))  # Remove duplicates

        if not scenarios:
            return None
        
        # Process each scenario
        for scen in scenarios:
            try:
                # Filter DataFrames for current scenario
                wlis_scen = wlis_df[wlis_df['Scenario'] == scen] if wlis_df is not None and 'Scenario' in wlis_df.columns else None
                daily_vars_scen = daily_vars_df[daily_vars_df['Scenario'] == scen] if daily_vars_df is not None and 'Scenario' in daily_vars_df.columns else None
                
                # Initialize variables for yearly summaries
                df_yearly_sum = None
                df_ao_summary = None
                
                # Process daily_vars_df (sum values grouped by year)
                if daily_vars_scen is not None:
                    daily_vars_scen['year'] = pd.to_datetime(daily_vars_scen['Date']).dt.year
                    df_yearly_sum = daily_vars_scen.groupby('year').sum(numeric_only=True)
                    df_yearly_sum.index.name = 'year'
                
                # Process wlis_df (aggregated_outputs logic)
                if wlis_scen is not None:
                    # Create a 'year' column from the 'time' column
                    wlis_scen['year'] = np.floor(wlis_scen['time']).astype(int)
                    
                    # Define columns to select for the summary
                    columns_to_select = list(wlis_scen.columns)
                    
                    # Get the index of rows with the maximum 'somsc' value for each year
                    idx = wlis_scen.groupby('year')['time'].idxmax()
                    
                    # Create a summary dataframe
                    df_ao_summary = wlis_scen.loc[idx, columns_to_select]
                    df_ao_summary.set_index('year', inplace=True)
                
                # Merge the DataFrames horizontally for this scenario
                combined_df_list = [df for df in [df_yearly_sum, df_ao_summary] if df is not None]
                
                if combined_df_list:
                    # Use outer join to ensure all indices are preserved
                    concat_df = pd.concat(combined_df_list, axis=1, join="outer")
                    
                    # Handle duplicate columns by keeping the last occurrence
                    if concat_df.columns.duplicated().any():
                        concat_df = concat_df.loc[:, ~concat_df.columns.duplicated(keep='last')]
                    
                    # Sort by index (year)
                    concat_df = concat_df.sort_index()
                    
                    # Add scenario column
                    concat_df['Scenario'] = scen
                    
                    # Append to list of processed dataframes (keeping year as index)
                    processed_dfs.append(concat_df)
                    
            except Exception as e:
                continue  # Skip this scenario if there's an error
        
        # Check if we have any successfully processed dataframes
        if not processed_dfs:
            return None
            
        # Combine all scenarios vertically if there are multiple scenarios
        if len(processed_dfs) >= 2:
            final_df = pd.concat(processed_dfs, axis=0)
            # Sort by year
            final_df.sort_index(inplace=True)
        else:
            final_df = processed_dfs[0]
        return final_df
        
    except Exception as e:
        return None


def calculate_advanced_statistics(df, var, metrics_options=None):
    """
    Calculate advanced statistics for time series data
    """
    if metrics_options is None:
        metrics_options = ['basic']
    
    stats_dict = {}
    series = df[var]
    
    # Basic statistics
    if 'basic' in metrics_options:
        stats_dict.update({
            'Maximum': series.max(),
            'Minimum': series.min(),
            'Mean': series.mean(),
            'Median': series.median(),
            'Standard Deviation': series.std(),
            'Variance': series.var()
        })
    
    # Distribution statistics
    if 'distribution' in metrics_options:
        stats_dict.update({
            'Skewness': series.skew(),
            'Kurtosis': series.kurtosis(),
            'Q1 (25th percentile)': series.quantile(0.25),
            'Q3 (75th percentile)': series.quantile(0.75),
            'IQR': series.quantile(0.75) - series.quantile(0.25)
        })
    
    # Time series specific metrics
    if 'time_series' in metrics_options and len(series) > 2:
        # Rolling statistics
        stats_dict.update({
            'Rolling Mean (7-day)': series.rolling(window=7).mean().iloc[-1],
            'Rolling Std (7-day)': series.rolling(window=7).std().iloc[-1]
        })
        
        # Growth rates
        pct_change = series.pct_change()
        stats_dict.update({
            'Average Growth Rate (%)': pct_change.mean() * 100,
            'Growth Rate Volatility (%)': pct_change.std() * 100
        })
        
        try:
            decomposition = seasonal_decompose(series, period=7, extrapolate_trend='freq')
            stats_dict.update({
                'Trend (Last Value)': decomposition.trend.iloc[-1],
                'Seasonal (Last Value)': decomposition.seasonal.iloc[-1]
            })
        except:
            pass
        
        try:
            adf_result = adfuller(series.dropna())
            stats_dict['ADF Statistic'] = adf_result[0]
            stats_dict['ADF p-value'] = adf_result[1]
        except:
            pass

    # Outlier detection
    if 'outliers' in metrics_options:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        stats_dict['Outlier Count'] = outlier_count
        
    return stats_dict

def plot_scenarios_streamlit_csv(dataframes_w, dataframes_dv, obs_data = None):
    st.sidebar.header("Plot Scenarios")

    # Merge dataframes with validation
    merged_w = pd.DataFrame()
    if dataframes_w:
        merged_w = pd.concat(dataframes_w, axis=0, join="outer", ignore_index=True)

    merged_dv = pd.DataFrame()
    if dataframes_dv:
        merged_dv = pd.concat(dataframes_dv, axis=0, join="outer", ignore_index=True)


    # Allow user to choose DataFrame
    available_dfs = {"Aggregated outputs DataFrame": merged_w, "Daily vars outputs Dataframe": merged_dv}
    available_dfs = {key: df for key, df in available_dfs.items() if not df.empty}

    if not available_dfs:
        st.error("No valid DataFrames to display. Please check your inputs.")
        return


    selected_df_label = st.sidebar.selectbox("Select DataFrame", list(available_dfs.keys()), label_visibility="collapsed")
    selected_df = available_dfs[selected_df_label]
    try:
        selected_df = process_time_column(selected_df=selected_df)
    except:
        pass

    if obs_data is not None:
        selected_df1 = pd.merge(selected_df, obs_data, on='time', how='outer')
    else:
        selected_df1 =selected_df
    # Filter by year range if 'time' column exists
    if 'time' in selected_df1.columns:
        min_year = int(selected_df1['time'].min())
        max_year = int(selected_df1['time'].max())
        year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
        selected_df1 = selected_df1[(selected_df1['time'] >= year_range[0]) & (selected_df1['time'] <= year_range[1])]

    
    # Variable selection
    st.markdown("### 📊 Select Variables to Plot")
    variables = [col for col in selected_df.columns if col not in ['Date', 'time', 'dayofyr', 'days_in_year', 'Scenario']]
    selected_vars = st.multiselect("Select variables to analyze:", variables)

    if not selected_vars:
        st.info("👆 Please select at least one variable to display the analysis.")
        return
    

    # Metrics selection
    metrics_groups = {'basic': 'Basic Statistics', 'time_series': 'Time Series Metrics'}
    st.sidebar.markdown("### 📈 Statistical Analysis Options")
    selected_metrics = st.sidebar.multiselect(
        "Select Statistical Metrics:",
        options=list(metrics_groups.keys()),
        default=['basic'],
        format_func=lambda x: metrics_groups[x],
    )


    # Analysis for each variable
    for column in selected_vars:
        filtered_df, selected_obs_columns, ID_select = data_transformation_ui(selected_df1, column, obs_data)
        # Analyze and visualize emissions
        visualize_scenario_trends(filtered_df, column=column, year_range=year_range, selected_metrics=selected_metrics, obs_op=selected_obs_columns, ID_select=ID_select)

def visualize_scenario_trends(selected_df, column, year_range, selected_metrics, obs_op=None, ID_select=None):
    """
    Visualize time series trends grouped by scenario, with optional scatter plots.

    Args:
        selected_df (pd.DataFrame): The filtered DataFrame containing data for analysis.
        obs_op (list, optional): Additional columns to plot as scatter plots.
        column (str, optional): The primary column to analyze as a line plot.
        year_range (tuple, optional): The selected time range for analysis.
        selected_metrics (list, optional): The metrics selected for statistical analysis.

    Returns:
        None
    """

    # Plot section
    st.markdown("### 📊 Trend Visualization")
    
    # Create figure with subplots if both line and scatter plots are needed
    # Create figure
    fig = go.Figure()

    # Add line traces for primary column
    if column:
        for scen in selected_df['Scenario'].unique():
            scen_df = selected_df[selected_df['Scenario'] == scen]
            fig.add_trace(
                go.Scatter(
                    x=scen_df['time'], 
                    y=scen_df[column], 
                    mode='lines+markers',
                    name=f'{scen}',
                    line=dict(width=2)
                )
            )

    # Add scatter traces for additional columns
    if obs_op:
        for obs in obs_op:
            for scen in ID_select:
                scen_df = selected_df[selected_df['ID'] == scen]
                fig.add_trace(
                    go.Scatter(
                        x=scen_df['time'], 
                        y=scen_df[obs], 
                        mode='markers',
                        name=f'{obs} - {scen}',
                        marker=dict(size=8, opacity=0.7)
                    )
                )

    # Update layout
    fig.update_layout(
        title=f'Trend of {column} (original unit) ({year_range[0]} - {year_range[1]})',
        xaxis_title='Time',
        yaxis_title='Values',
        template="plotly_dark",
        legend_title="Scenario"
    )

    # Update traces with legend info in hovertemplate
    fig.update_traces(
        hovertemplate="<b>Time: %{x}</b><br>" +
                    "Scenario: %{fullData.name}<br>" +
                    "Value: %{y}<extra></extra>"
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Rest of the function remains the same as in the original implementation
    # (statistical analysis and time series components section)
    
    # Statistical analysis section (only if metrics are provided)
    if selected_metrics:
        st.markdown("### 📊 Statistical Analysis")
    
        # Exclude NaN values from Scenario column
        unique_scenarios = selected_df['Scenario'].dropna().unique()
        
        for scen in unique_scenarios:
            st.markdown(f"#### Scenario: {scen}")
            scen_df = selected_df[selected_df['Scenario'] == scen]
            
            # Analyze primary column if exists
            if column:
                stats_results = calculate_time_series_metrics(scen_df[column], selected_metrics)
                
                # Display metrics
                metrics_per_row = 5
                metrics_items = list(stats_results.items())

                for i in range(0, len(metrics_items), metrics_per_row):
                    cols = st.columns(metrics_per_row)
                    for j in range(metrics_per_row):
                        if i + j < len(metrics_items):
                            metric, value = metrics_items[i + j]
                            with cols[j]:
                                try:
                                    formatted_value = f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
                                    st.markdown(f"<div style='font-size: 12px;'><b>{metric}:</b> {formatted_value}</div>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown(f"<div style='font-size: 12px;'>Error: {str(e)}</div>", unsafe_allow_html=True)
            
            # Time series components visualization
        if 'time_series' in selected_metrics:
            st.sidebar.markdown("### 🌦️ Seasonal Period Settings")
            for scen in selected_df['Scenario'].unique():
                seasonal_period_key = f"seasonal_period_slider_{scen}_{column}"
                seasonal_period = st.sidebar.slider(
                    "Select Seasonal Period (e.g., 12 for monthly, 4 for quarterly):",
                    min_value=1, max_value=24, value=12,
                    key=seasonal_period_key
                )

                st.markdown(f"### 📈 Time Series Components for Scenario: {scen}")
                scen_df = selected_df[selected_df['Scenario'] == scen]

                # Handle missing values
                time_series = scen_df[column].dropna()  # Remove NaNs from the time series

                if time_series.empty:
                    st.warning(f"No data available for Scenario: {scen} after removing missing values.")
                    continue

                try:
                    decomposition = seasonal_decompose(
                        time_series,
                        period=seasonal_period,
                        extrapolate_trend='freq'
                    )

                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Trend', 'Seasonal', 'Residual'),
                        vertical_spacing=0.1
                    )

                    # Add traces
                    fig.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'),
                        row=3, col=1
                    )

                    fig.update_layout(height=800, title_text=f"Time Series Decomposition - Scenario: {scen}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not perform seasonal decomposition for Scenario: {scen}. Error: {e}")


            st.markdown("---")  # Separator between variables





def plot_scenarios_streamlit_csv2(dataframes_combine, dataframes_ao, dataframes_harvest, obs_data=None):
    st.sidebar.header("Plot Scenarios")

    # Merge dataframes with validation
    merged_combine = pd.DataFrame()
    if dataframes_combine:
        merged_combine = pd.concat(dataframes_combine, axis=0, join="outer", ignore_index=True)

    merged_ao = pd.DataFrame()
    if dataframes_ao:
        merged_ao = pd.concat(dataframes_ao, axis=0, join="outer", ignore_index=True)

    merged_harvest = pd.DataFrame()
    if dataframes_harvest:
        merged_harvest = pd.concat(dataframes_harvest, axis=0, join="outer", ignore_index=True)

    # Allow user to choose DataFrame
    available_dfs = {
        "Daily vars DataFrame": merged_combine,
        "Aggregated outputs DataFrame": merged_ao,
        "Harvest DataFrame": merged_harvest
    }
    available_dfs = {key: df for key, df in available_dfs.items() if not df.empty}

    if not available_dfs:
        st.error("No valid DataFrames to display. Please check your inputs.")
        return

    selected_df_label = st.sidebar.selectbox("Select DataFrame", list(available_dfs.keys()), label_visibility="collapsed")
    selected_df = available_dfs[selected_df_label]

    try:
        selected_df = process_time_column(selected_df=selected_df)
    except:
        pass

    if obs_data is not None:
        selected_df1 = pd.merge(selected_df, obs_data, on='time', how='outer')
    else:
        selected_df1 =selected_df
    # Filter by year range if 'time' column exists
    if 'time' in selected_df1.columns:
        min_year = int(selected_df1['time'].min())
        max_year = int(selected_df1['time'].max())
        year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
        selected_df1 = selected_df1[(selected_df1['time'] >= year_range[0]) & (selected_df1['time'] <= year_range[1])]

    
    # Variable selection
    st.markdown("### 📊 Select Variables to Plot")
    variables = [col for col in selected_df.columns if col not in ['Date', 'time', 'dayofyr', 'days_in_year', 'Scenario']]
    selected_vars = st.multiselect("Select variables to analyze:", variables)

    if not selected_vars:
        st.info("👆 Please select at least one variable to display the analysis.")
        return
    

    # Metrics selection
    metrics_groups = {'basic': 'Basic Statistics', 'time_series': 'Time Series Metrics'}
    st.sidebar.markdown("### 📈 Statistical Analysis Options")
    selected_metrics = st.sidebar.multiselect(
        "Select Statistical Metrics:",
        options=list(metrics_groups.keys()),
        default=['basic'],
        format_func=lambda x: metrics_groups[x],
    )


    # Analysis for each variable
    for column in selected_vars:
        filtered_df, selected_obs_columns, ID_select = data_transformation_ui(selected_df1, column, obs_data)
        # Analyze and visualize emissions
        visualize_scenario_trends(filtered_df, column=column, year_range=year_range, selected_metrics=selected_metrics, obs_op=selected_obs_columns, ID_select=ID_select)
        # Analyze and visualize emissions
        #visualize_scenario_trends(selected_df1, column=column, year_range=year_range, selected_metrics=selected_metrics, obs_op=selected_obs_columns, ID_select=ID_select)


# Initialize DuckDB connection
@st.cache_resource
def init_duckdb():
    """Initialize DuckDB connection and return it"""
    conn = duckdb.connect()
    return conn

def setup_duckdb_tables(conn, dataframes_combine, dataframes_ao, dataframes_harvest, obs_data=None):
    """
    Setup DuckDB tables from dataframes
    
    Args:
        conn: DuckDB connection
        dataframes_combine: List of combined dataframes
        dataframes_ao: List of aggregated output dataframes  
        dataframes_harvest: List of harvest dataframes
        obs_data: Observation data (optional)
    """
    
    # Create tables for combined data
    if dataframes_combine:
        merged_combine = pd.concat(dataframes_combine, axis=0, join="outer", ignore_index=True)
        conn.execute("DROP TABLE IF EXISTS daily_vars")
        conn.execute("CREATE TABLE daily_vars AS SELECT * FROM merged_combine")
    
    # Create tables for aggregated outputs
    if dataframes_ao:
        merged_ao = pd.concat(dataframes_ao, axis=0, join="outer", ignore_index=True)
        conn.execute("DROP TABLE IF EXISTS aggregated_outputs")
        conn.execute("CREATE TABLE aggregated_outputs AS SELECT * FROM merged_ao")
    
    # Create tables for harvest data
    if dataframes_harvest:
        merged_harvest = pd.concat(dataframes_harvest, axis=0, join="outer", ignore_index=True)
        conn.execute("DROP TABLE IF EXISTS harvest_data")
        conn.execute("CREATE TABLE harvest_data AS SELECT * FROM merged_harvest")
    
    # Create observation data table if provided
    if obs_data is not None:
        conn.execute("DROP TABLE IF EXISTS obs_data")
        conn.execute("CREATE TABLE obs_data AS SELECT * FROM obs_data")

def get_available_tables(conn):
    """Get list of available tables in DuckDB"""
    result = conn.execute("SHOW TABLES").fetchall()
    return [table[0] for table in result]

def get_table_columns(conn, table_name):
    """Get columns of a specific table"""
    result = conn.execute(f"DESCRIBE {table_name}").fetchall()
    return [col[0] for col in result]

def filter_data_by_year_range(conn, table_name, year_range):
    """
    Filter data by year range using DuckDB
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to filter
        year_range: Tuple of (min_year, max_year)
    
    Returns:
        Filtered dataframe
    """
    query = f"""
    SELECT * FROM {table_name} 
    WHERE time >= {year_range[0]} AND time <= {year_range[1]}
    """
    return conn.execute(query).df()

def get_year_range_from_table(conn, table_name):
    """Get min and max years from a table"""
    query = f"SELECT MIN(time) as min_year, MAX(time) as max_year FROM {table_name}"
    result = conn.execute(query).fetchone()
    return int(result[0]), int(result[1])

def get_unique_scenarios(conn, table_name):
    """Get unique scenarios from a table"""
    query = f"SELECT DISTINCT Scenario FROM {table_name} WHERE Scenario IS NOT NULL"
    result = conn.execute(query).fetchall()
    return [row[0] for row in result]

def get_scenario_data(conn, table_name, scenario, column=None, year_range=None):
    """
    Get data for a specific scenario
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table
        scenario: Scenario name
        column: Specific column to select (optional)
        year_range: Year range filter (optional)
    
    Returns:
        Filtered dataframe
    """
    # Properly quote column names to handle special characters and parentheses
    if column:
        # Use double quotes to properly escape column names with special characters
        select_clause = f'time, "{column}"'
    else:
        select_clause = "*"
    
    where_clauses = [f"Scenario = '{scenario}'"]
    
    if year_range:
        where_clauses.append(f"time >= {year_range[0]} AND time <= {year_range[1]}")
    
    query = f"""
    SELECT {select_clause} FROM {table_name} 
    WHERE {' AND '.join(where_clauses)}
    ORDER BY time
    """
    return conn.execute(query).df()

def merge_with_obs_data(conn, main_table, obs_table=None):
    """
    Merge main table with observation data using DuckDB
    
    Args:
        conn: DuckDB connection
        main_table: Name of main table
        obs_table: Name of observation table (optional)
    
    Returns:
        Merged dataframe
    """
    if obs_table and obs_table in get_available_tables(conn):
        query = f"""
        SELECT * FROM {main_table} m
        FULL OUTER JOIN {obs_table} o ON m.time = o.time
        """
        return conn.execute(query).df()
    else:
        return conn.execute(f"SELECT * FROM {main_table}").df()

def calculate_statistics_duckdb(conn, table_name, column, scenario=None):
    """
    Calculate basic statistics using DuckDB
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table
        column: Column to analyze
        scenario: Specific scenario (optional)
    
    Returns:
        Dictionary of statistics
    """
    where_clause = f"WHERE Scenario = '{scenario}'" if scenario else ""
    
    query = f"""
    SELECT 
        COUNT({column}) as count,
        AVG({column}) as mean,
        STDDEV({column}) as std,
        MIN({column}) as min,
        MAX({column}) as max,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as q75
    FROM {table_name}
    {where_clause}
    """
    
    result = conn.execute(query).fetchone()
    return {
        'Count': result[0],
        'Mean': result[1],
        'Std': result[2],
        'Min': result[3],
        'Max': result[4],
        'Q25': result[5],
        'Median': result[6],
        'Q75': result[7]
    }

def visualize_scenario_trends_duckdb(conn, table_name, column, year_range, selected_metrics, obs_columns=None, ID_select=None):
    """
    Visualize time series trends using DuckDB for data processing
    
    Args:
        conn: DuckDB connection
        table_name: Name of the table to analyze
        column: Primary column to analyze
        year_range: Year range for filtering
        selected_metrics: Selected statistical metrics
        obs_columns: Additional observation columns
        ID_select: Selected IDs for observation data
    """
    try:
        st.markdown("### 📊 Trend Visualization")
        
        # Create figure
        fig = go.Figure()
        
        # Define color palette with enough distinct colors
        # Define extended color palette with 50 distinct colors
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5',
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
            '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
            '#6c5ce7', '#fd79a8', '#fdcb6e', '#e84393', '#00b894',
            '#0984e3', '#a29bfe', '#fd79a8', '#e17055', '#00cec9',
            '#6c5ce7', '#fab1a0', '#00b894', '#74b9ff', '#fd79a8',
            '#fdcb6e', '#e84393', '#0984e3', '#a29bfe', '#e17055'
        ]
        
        
        # Get unique scenarios
        scenarios = get_unique_scenarios(conn, table_name)
        
        # Create color mapping for scenarios
        scenario_colors = {}
        for i, scenario in enumerate(scenarios):
            scenario_colors[scenario] = color_palette[i % len(color_palette)]
        
        # Add line traces for primary column
        if column:
            for i, scenario in enumerate(scenarios):
                # Get scenario data using DuckDB
                scen_df = get_scenario_data(conn, table_name, scenario, column, year_range)
                
                if not scen_df.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=scen_df['time'], 
                            y=scen_df[column], 
                            mode='lines+markers',
                            name=f'{scenario}',
                            line=dict(width=2, color=scenario_colors[scenario]),
                            marker=dict(color=scenario_colors[scenario])
                        )
                    )
        
        # Add scatter traces for observation columns
        if obs_columns and ID_select:
            # Create color mapping for observations (using different shades/patterns)
            obs_color_start = len(scenarios)
            
            for obs_idx, obs in enumerate(obs_columns):
                for id_idx, scenario_id in enumerate(ID_select):
                    # Query for observation data
                    query = f"""
                    SELECT time, {obs} FROM {table_name}
                    WHERE ID = '{scenario_id}' 
                    AND time >= {year_range[0]} AND time <= {year_range[1]}
                    ORDER BY time
                    """
                    obs_df = conn.execute(query).df()
                    
                    if not obs_df.empty:
                        # Use different color for observations
                        color_idx = (obs_color_start + obs_idx * len(ID_select) + id_idx) % len(color_palette)
                        obs_color = color_palette[color_idx]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=obs_df['time'], 
                                y=obs_df[obs], 
                                mode='markers',
                                name=f'{obs} - {scenario_id}',
                                marker=dict(
                                    size=8, 
                                    opacity=0.7, 
                                    color=obs_color,
                                    symbol='diamond'  # Different symbol for observations
                                )
                            )
                        )
        
        # Update layout
        fig.update_layout(
            title=f'Trend of {column} (original unit) ({year_range[0]} - {year_range[1]})',
            xaxis_title='Time',
            yaxis_title='Values',
            template="plotly_dark",
            legend_title="Scenario",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Update traces with legend info in hovertemplate
        fig.update_traces(
            hovertemplate="<b>Time: %{x}</b><br>" +
                        "Scenario: %{fullData.name}<br>" +
                        "Value: %{y}<extra></extra>"
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Checkbox to control statistical analysis display
        show_analysis = st.checkbox(
            f"📊 Show Statistical Analysis for {column}", 
            value=False,
            key=f"show_analysis_{column}_{table_name}"
        )
        
        # Statistical analysis section (only show if checkbox is checked)
        if show_analysis and selected_metrics:
            st.markdown("### 📊 Statistical Analysis")
            
            for scenario in scenarios:
                with st.expander(f"📈 Statistics for Scenario: {scenario}", expanded=False):
                    # Calculate statistics using DuckDB
                    if column and 'basic' in selected_metrics:
                        stats_results = calculate_statistics_duckdb(conn, table_name, column, scenario)
                        
                        # Display metrics in a more compact way
                        metrics_per_row = 4
                        metrics_items = list(stats_results.items())
                        
                        for i in range(0, len(metrics_items), metrics_per_row):
                            cols = st.columns(metrics_per_row)
                            for j in range(metrics_per_row):
                                if i + j < len(metrics_items):
                                    metric, value = metrics_items[i + j]
                                    with cols[j]:
                                        try:
                                            formatted_value = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                                            st.metric(label=metric, value=formatted_value)
                                        except Exception as e:
                                            st.error(f"Error calculating {metric}: {str(e)}")
            
            # Time series components visualization
            if 'time_series' in selected_metrics:
                st.markdown("### 📈 Time Series Decomposition")
                
                # Sidebar settings for seasonal period
                st.sidebar.markdown("### 🌦️ Seasonal Period Settings")  
                seasonal_period = st.sidebar.slider(
                    f"Seasonal Period for {column} (e.g., 12 for monthly, 4 for quarterly):",
                    min_value=1, max_value=24, value=12,
                    key=f"seasonal_period_{column}_{table_name}"
                )
                
                # Select scenarios for time series analysis
                selected_scenarios = st.multiselect(
                    "Select scenarios for time series decomposition:",
                    options=scenarios,
                    default=scenarios[:2] if len(scenarios) >= 2 else scenarios,
                    key=f"ts_scenarios_{column}_{table_name}"
                )
                
                for scenario in selected_scenarios:
                    with st.expander(f"📈 Time Series Components for Scenario: {scenario}", expanded=False):
                        # Get scenario data for time series analysis
                        scen_df = get_scenario_data(conn, table_name, scenario, column, year_range)
                        
                        if scen_df.empty:
                            st.warning(f"No data available for Scenario: {scenario}")
                            continue
                        
                        # Handle missing values
                        time_series = scen_df[column].dropna()
                        
                        if time_series.empty:
                            st.warning(f"No data available for Scenario: {scenario} after removing missing values.")
                            continue
                        
                        try:
                            decomposition = seasonal_decompose(
                                time_series,
                                period=seasonal_period,
                                extrapolate_trend='freq'
                            )
                            
                            # Create subplots with consistent colors
                            fig_ts = make_subplots(
                                rows=3, cols=1,
                                subplot_titles=('Trend', 'Seasonal', 'Residual'),
                                vertical_spacing=0.1
                            )
                            
                            # Use scenario color for decomposition
                            scenario_color = scenario_colors[scenario]
                            
                            # Add traces
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=decomposition.trend.index, 
                                    y=decomposition.trend, 
                                    name='Trend',
                                    line=dict(color=scenario_color)
                                ),
                                row=1, col=1
                            )
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=decomposition.seasonal.index, 
                                    y=decomposition.seasonal, 
                                    name='Seasonal',
                                    line=dict(color=scenario_color, dash='dash')
                                ),
                                row=2, col=1
                            )
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=decomposition.resid.index, 
                                    y=decomposition.resid, 
                                    name='Residual',
                                    line=dict(color=scenario_color, dash='dot')
                                ),
                                row=3, col=1
                            )
                            
                            fig_ts.update_layout(
                                height=600, 
                                title_text=f"Time Series Decomposition - {scenario}",
                                showlegend=False
                            )
                            st.plotly_chart(fig_ts, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Could not perform seasonal decomposition for Scenario: {scenario}. Error: {e}")
        
        st.markdown("---")
    except Exception as e:
        st.error(f"❌ Error displaying time series decomposition plot for Scenario {scenario}: {e}")

def plot_scenarios_streamlit_duckdb(all_daily_data, dataframes_ao, dataframes_harvest, obs_data=None):
    """
    Main function to plot scenarios using DuckDB for data processing
    
    Args:
        all_daily_data: List of tuples (list_daily_dfs, daily_keys) for each file
        dataframes_ao: List of aggregated output dataframes
        dataframes_harvest: List of harvest dataframes
        obs_data: Observation data (optional)
    """
    
    st.sidebar.header("Plot Scenarios")
    
    # Initialize DuckDB connection
    conn = init_duckdb()
    
    # Create mapping for user-friendly names
    table_mapping = {
        "daily_vars": "Daily vars DataFrame",
        "aggregated_outputs": "Aggregated outputs DataFrame", 
        "harvest_data": "Harvest DataFrame"
    }
    
    # Determine available tables
    available_tables = []
    if all_daily_data:
        # Check if any daily data has non-empty dataframes
        has_daily_data = any(
            daily_data[0] and len(daily_data[0]) > 0 
            for daily_data in all_daily_data 
            if daily_data and len(daily_data) == 2
        )
        if has_daily_data:
            available_tables.append("daily_vars")
    
    if dataframes_ao:
        available_tables.append("aggregated_outputs")
    if dataframes_harvest:
        available_tables.append("harvest_data")
    
    if not available_tables:
        st.error("No valid DataFrames to display. Please check your inputs.")
        return
    
    # Filter available tables and create display options
    display_options = {table_mapping.get(table, table): table for table in available_tables}
    
    # Let user select table
    selected_display_name = st.sidebar.selectbox("Select DataFrame", list(display_options.keys()))
    selected_table = display_options[selected_display_name]
    
    # If daily vars is selected, let user choose keys
    selected_keys = None
    if selected_table == "daily_vars":
        # Get all unique keys from all files
        all_keys = set()
        for daily_data in all_daily_data:
            if daily_data and len(daily_data) == 2:
                list_daily_dfs, daily_keys = daily_data
                if daily_keys:
                    all_keys.update(daily_keys)
        
        all_keys = sorted(list(all_keys))
        
        if not all_keys:
            st.error("No daily data keys available.")
            return
        
        st.sidebar.markdown("### 🔑 Select Daily Data Keys")
        selected_keys = st.sidebar.multiselect(
            "Choose which daily data keys to analyze:",
            options=all_keys,
            default=None
        )
        
        if not selected_keys:
            st.info("👆 Please select at least one daily data key to continue.")
            return
    
    # Setup DuckDB tables based on selection
    if selected_table == "daily_vars":
        # Only setup daily vars table with selected keys
        dataframes_combine = []
        for daily_data in all_daily_data:
            if daily_data and len(daily_data) == 2:
                list_daily_dfs, daily_keys = daily_data
                if list_daily_dfs and daily_keys:
                    for i, key in enumerate(daily_keys):
                        if key in selected_keys and i < len(list_daily_dfs):
                            df = list_daily_dfs[i].copy()
                            df['DataKey'] = key  # Add key identifier
                            dataframes_combine.append(df)
        
        setup_duckdb_tables(conn, dataframes_combine, [], [], obs_data)
    else:
        # Setup other tables
        if selected_table == "aggregated_outputs":
            setup_duckdb_tables(conn, [], dataframes_ao, [], obs_data)
        elif selected_table == "harvest_data":
            setup_duckdb_tables(conn, [], [], dataframes_harvest, obs_data)
    
    # Get year range for filtering
    try:
        min_year, max_year = get_year_range_from_table(conn, selected_table)
        year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
    except:
        year_range = (2000, 2030)  # Default range
    
    # Get filtered data with observation data merged
    if obs_data is not None:
        selected_df = merge_with_obs_data(conn, selected_table, "obs_data")
    else:
        selected_df = filter_data_by_year_range(conn, selected_table, year_range)
    
    # Variable selection
    st.markdown("### 📊 Select Variables to Plot")
    
    # Get table columns excluding system columns
    all_columns = get_table_columns(conn, selected_table)
    exclude_columns = ['Date', 'time', 'dayofyr', 'days_in_year', 'Scenario']
    if selected_table == "daily_vars":
        exclude_columns.append('DataKey')
    
    variables = [col for col in all_columns if col not in exclude_columns]
    
    selected_vars = st.multiselect("Select variables to analyze:", variables)
    
    if not selected_vars:
        st.info("👆 Please select at least one variable to display the analysis.")
        return
    
    # Show selected keys info for daily vars
    if selected_table == "daily_vars" and selected_keys:
        st.info(f"📊 Analyzing daily data for keys: {', '.join(selected_keys)}")
    
    # Metrics selection
    metrics_groups = {'basic': 'Basic Statistics', 'time_series': 'Time Series Metrics'}
    st.sidebar.markdown("### 📈 Statistical Analysis Options")
    selected_metrics = st.sidebar.multiselect(
        "Select Statistical Metrics:",
        options=list(metrics_groups.keys()),
        default=['basic'],
        format_func=lambda x: metrics_groups[x],
    )
    
    # Analysis for each variable
    for column in selected_vars:
        # Note: You'll need to implement data_transformation_ui_duckdb for DuckDB
        # For now, using simplified approach
        obs_columns = None
        ID_select = None
        
        # Analyze and visualize using DuckDB
        visualize_scenario_trends_duckdb(
            conn, 
            table_name=selected_table,
            column=column, 
            year_range=year_range, 
            selected_metrics=selected_metrics, 
            obs_columns=obs_columns, 
            ID_select=ID_select
        )



def load_paths_from_file(file_path):
    """
    Load file paths from a .txt or .json file
    Returns a list of paths to JSON files
    """
    if not file_path:
        return []
        
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:  # .txt file
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def load_paths_from_folder(folder_path):
    """
    Load all .json file paths from a given folder.
    Returns a list of paths to JSON files.
    """
    if not folder_path:
        return []

    try:
        if not os.path.isdir(folder_path):
            print(f"[Warning] The folder path '{folder_path}' is not valid.")
            return []

        json_files = glob.glob(os.path.join(folder_path, '*.json'))
        return json_files

    except Exception as e:
        print(f"Error reading folder {folder_path}: {str(e)}")
        return []

class BaselineProcessor:
    def __init__(self, df):
        """
        Initialize with a DataFrame that must contain at least the columns:
        'Scenario', 'location', and the column to be processed.
        
        Parameters:
            df (DataFrame): The input DataFrame.
        """
        self.df = df.copy()

    @staticmethod
    def get_baseline_value(df, crop_key, column_selected='Total_Soil_GHG (MgCO2e/ha)', base_content=["noRI_furr_wR0_nocc", "noRI_furr_wR90_nocc"]):
        """
        Get the baseline value for a given crop key, supporting multiple baseline scenarios.
        
        Parameters:
            df (DataFrame): DataFrame containing the data.
            crop_key (str): Crop identifier (e.g., "ct-m_").
            column_selected (str): Column name to extract the value from.
            base_content (list or str): List of possible baseline suffixes.
            
        Returns:
            float or None: The first found baseline value, or None if no baseline is found.
        """
        if isinstance(base_content, str):
            base_content = [base_content]  # Convert to list if a single string is given

        for base in base_content:
            baseline_scenario = crop_key + base
            baseline_index = df.index[df['Scenario'] == baseline_scenario].tolist()

            if baseline_index:
                return df.loc[baseline_index[0], column_selected]  # Return the first found value

        return None  # No baseline scenario found

    def process_locations_and_baselines(self, column_selected, base_content=["noRI_furr_wR0_nocc", "noRI_furr_wR90_nocc"]):
        """
        Process each location to calculate baseline values for each row and create a new DataFrame.
        
        For each unique location, it calculates the baseline value (using the crop_key extracted from the 'Scenario'
        column) and assigns it to a new column 'Baseline'.
        
        Parameters:
            column_selected (str): The column from which to calculate baseline values.
            base_content (list or str): List of possible baseline suffixes.
            
        Returns:
            DataFrame: A new DataFrame with a 'Baseline' column added.
        """
        dfs_result = []

        for location in self.df['location'].unique():
            location_df = self.df[self.df['location'] == location].copy()

            for idx, row in location_df.iterrows():
                scenario = row["Scenario"]
                splits = scenario.split("_")
                crop_key = splits[0] + "_"  # Assume crop_key is the first part with an underscore appended
                
                # Get the baseline value for this crop_key at the current location
                baseline_value = BaselineProcessor.get_baseline_value(location_df, crop_key, 
                                                                      column_selected=column_selected, 
                                                                      base_content=base_content)
                location_df.loc[idx, 'Baseline'] = baseline_value
            
            dfs_result.append(location_df)

        merged_df = pd.concat(dfs_result, ignore_index=True)
        return merged_df

    def relative_different(self, selected_column='Total_Soil_GHG (MgCO2e/ha)', base_content=["noRI_furr_wR0_nocc", "noRI_furr_wR90_nocc"]):
        """
        Calculate the relative difference as: selected_column - Baseline.
        
        This method processes each location to add a 'Baseline' column and then computes a new column 
        'Emission Reduction (MgCO2e/ha)'.
        
        Parameters:
            selected_column (str): The column used for calculation (default is 'Total_Soil_GHG (MgCO2e/ha)').
            base_content (list or str): List of possible baseline suffixes.
        
        Returns:
            DataFrame: The DataFrame with the new 'Emission Reduction (MgCO2e/ha)' column.
        """
        merged_df = self.process_locations_and_baselines(selected_column, base_content)
        merged_df['Emission Reduction (MgCO2e/ha)'] = merged_df[selected_column] - merged_df['Baseline']
        return merged_df



def box_plot(df_raw, baseline_content=None):
    """
    Creates interactive boxplots with enhanced hover information using Plotly Express for Streamlit.
    The function allows users to select multiple columns to plot.

    Parameters:
        dfs (list of pandas.DataFrame): List of DataFrames containing the data.
        baseline_content (list or str or None, optional): List of strings or single string to identify baseline scenarios.
                                                         If a scenario name contains any string in this list,
                                                         it will be plotted as a baseline.
    """

    df = df_raw.copy()

    # Retrieve numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Allow users to select multiple columns for boxplots
    selected_columns = st.multiselect(
        "Select columns to plot",
        options=numeric_columns,
        default=None if numeric_columns else None
    )

    if not selected_columns:
        st.info("Please select at least one column to plot.")
        return

    # Get all scenarios and sort them to maintain consistent order
    all_scenarios = sorted(df['Scenario'].dropna().unique().tolist())

    # Identify baseline scenarios if applicable
    baseline_scenarios = []
    if baseline_content:
        # Convert to list if it's a string
        if isinstance(baseline_content, str):
            baseline_content = [baseline_content]
        
        # Check if any scenario contains any string in the baseline_content list
        baseline_scenarios = [
            sc for sc in all_scenarios 
            if any(content.lower() in str(sc).lower() for content in baseline_content)
        ]

        if not baseline_scenarios:
            st.info(f"No scenarios containing any of the specified baseline content found. All scenarios will be shown in the same color.")
            baseline_content = None
        else:
            # Mark scenarios as baseline or treatment
            df['is_baseline'] = df['Scenario'].apply(
                lambda x: 'Baseline' if any(content.lower() in str(x).lower() for content in baseline_content) else 'Treatment'
            )

    # Process each selected column
    for column_selected in selected_columns:
        # Compute summary statistics for the selected column
        stats_df = df.groupby('Scenario')[column_selected].agg(
            Mean="mean",
            Median="median",
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Min="min",
            Max="max",
            Count="count"
        ).round(2).reset_index()

        # Merge statistics back into the DataFrame
        temp_df = df.merge(stats_df, on="Scenario", how="left")

        # Create a boxplot using Plotly
        fig = px.box(
            temp_df,
            x="Scenario",
            y=column_selected,
            color="is_baseline" if baseline_content else None,
            color_discrete_map={"Baseline": "rgba(31, 119, 180, 0.8)", "Treatment": "rgba(255, 127, 14, 0.8)"},
            title=f"Boxplot of {column_selected} by Treatment (Scenario)",
            labels={"Scenario": "Treatment (Scenario)", column_selected: f"{column_selected}"},
            template="plotly_white",
            points="outliers",
            hover_data=["Mean", "Median", "Q1", "Q3", "Min", "Max", "Count"],
            category_orders={"Scenario": all_scenarios}  # Ensure consistent category order
        )

        # Adjust y-axis range dynamically
        y_min = temp_df[column_selected].min()
        y_max = temp_df[column_selected].max()
        y_range_buffer = max(abs(y_max - y_min) * 0.2, 0.5)
        fig.update_yaxes(range=[y_min - y_range_buffer, y_max + y_range_buffer])

        # Improve layout and appearance
        fig.update_layout(
            xaxis_tickangle=-90,  # Less extreme angle for better readability
            plot_bgcolor="white",
            margin=dict(l=60, r=40, t=60, b=100),  # Increased margins for better spacing
            xaxis=dict(
                title_font=dict(size=16, family="Arial", color="black"),
                tickfont=dict(size=14, family="Arial", color="black"),
                showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)",
                zeroline=True, zerolinewidth=2, mirror=True, showline=True, linewidth=2, linecolor="black",
                categoryarray=all_scenarios  # Ensure categories are in the correct order
            ),
            yaxis=dict(
                title_font=dict(size=16, family="Arial", color="black"),
                tickfont=dict(size=14, family="Arial", color="black"),
                showgrid=True, gridcolor="rgba(200, 200, 200, 0.3)",
                zeroline=True, zerolinewidth=2, mirror=True, showline=True, linewidth=2, linecolor="black"
            ),
            legend=dict(
                orientation="h",
                title=None,
                y=-2.,  # Moved up a bit from the bottom
                x=0.5,
                xanchor="center",
                font=dict(size=16, family="Arial", color="black"),
                bgcolor="rgba(255, 255, 255, 0.9)",  # Light background for better visibility
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                itemsizing="constant",
                itemwidth=30,
                itemclick=False,
                traceorder="normal",
                tracegroupgap=10,
                yanchor="bottom"
            ),
            title=dict(font_size=18, font_family="Arial", x=0.5, xanchor="center")
        )

        # Add a horizontal reference line at y = 0
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(temp_df['Scenario'].unique()) - 0.5,
            y0=0, y1=0,
            line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash")
        )

        # Display the plot in Streamlit
        st.subheader(f"Boxplot for {column_selected}")
        st.plotly_chart(fig, use_container_width=True)


def plot_relative_difference(
    df_merged,
    baseline_content=["noRI_furr_wR0_nocc", "noRI_furr_wR90_nocc"],
    factor_list=None
):
    """
    Create a box plot of emission reduction across different scenarios and display it in Streamlit.
    Parameters:
    -----------
    df_merged : pandas.DataFrame
        Input DataFrame containing scenario and emission reduction data
    baseline_content : list or str, optional
        Scenarios to exclude as baseline (default: noRI scenarios)
    factor_list : list, optional
        Specific factors to filter scenarios
    """
    # Ensure baseline_content is a list
    if isinstance(baseline_content, str):
        baseline_content = [baseline_content]
    # Create a deep copy to avoid modifying original DataFrame
    df_working = df_merged.copy()
    # Initialize factor_list if None
    factor_list = factor_list or []
    # Prepare search factors based on input
    if "cc" in factor_list or "nocc" in factor_list:
        search_factors = [
            '_' + f for f in factor_list
        ] + [
            "_" + f + "_" for f in factor_list
        ] + [
            f + "_" for f in factor_list
        ]
    else:
        search_factors = [
            "_" + f + "_" for f in factor_list
        ] + [
            f + "_" for f in factor_list
        ] + [
            '_' + f for f in factor_list
        ]
    # Filter out baseline scenarios
    df_filtered = df_working[
        ~df_working['Scenario'].apply(
            lambda x: any(base in x for base in baseline_content)
        )
    ]
    # Filter scenarios based on specified factors
    if search_factors:
        df_filtered = df_filtered[
            df_filtered['Scenario'].apply(
                lambda x: any(factor in x for factor in search_factors)
            )
        ]
    # Handling empty DataFrame after filtering
    if df_filtered.empty:
        st.error("No data available after filtering with the given factor list.")
        return
    # Extract factors for legend labeling
    def extract_factors(x, search_factors):
        for factor in search_factors:
            if factor in x:
                return factor
        return x  # Return full scenario name if no factor matches
    # Create legend labels
    df_filtered["Legend_Label"] = df_filtered["Scenario"].apply(
        lambda x: extract_factors(x, search_factors)
    )
    df_filtered["Legend_Label"] = df_filtered["Legend_Label"].str.replace("_", "", regex=False)
    # Optional: Map factors using a predefined mapping (if available)
    try:
        if hasattr(Constants, "factor_mapping"):
            df_filtered["Legend_Label"] = df_filtered["Legend_Label"].map(
                Constants.factor_mapping
            ).fillna(df_filtered["Legend_Label"])
    except NameError:
        pass  # Constants class not defined, skip this step
    # Compute statistical summary
    stats_df = df_filtered.groupby('Scenario')['Emission Reduction (MgCO2e/ha)'].agg(
        Mean="mean",
        Median="median",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
        Min="min",
        Max="max",
        Count="count"
    ).round(2).reset_index()
    # Merge statistical summary back to filtered DataFrame
    df_filtered = df_filtered.merge(stats_df, on='Scenario', how='left')
    # Create box plot with enhanced styling
    fig = px.box(
        df_filtered,
        x='Scenario',
        y='Emission Reduction (MgCO2e/ha)',
        color='Legend_Label',
        title="Emission Reduction (MgCO2e/ha) by Scenario",
        labels={
            "Scenario": "Scenario",
            "Emission Reduction (MgCO2e/ha)": "Emission Reduction (MgCO2e/ha)",
            "Legend_Label": "Scenario Factors"
        },
        template="plotly_white",
        points="outliers",
        hover_data=['Mean', 'Median', 'Q1', 'Q3', 'Min', 'Max', 'Count'],
        category_orders={'Scenario': df_filtered['Scenario'].unique()}  # Preserve original order
    )
    # Update traces to improve box plot appearance
    fig.update_traces(
        width=0.8,  # Increases box width
        boxpoints='outliers'  # Shows outliers
    )
    # Compute y-axis range with buffer
    y_min = df_filtered['Emission Reduction (MgCO2e/ha)'].min()
    y_max = df_filtered['Emission Reduction (MgCO2e/ha)'].max()
    y_range_buffer = max(abs(y_max - y_min) * 0.2, 0.5)
    # Enhanced layout configuration
    fig.update_layout(
        xaxis_tickangle=-90,  # More readable tick angle
        plot_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=100),  # Increased margins
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.3)",
            gridwidth=1,
            title_font=dict(size=12),
            tickfont=dict(size=12, family="Arial", color="black"),
            zeroline=True,
            zerolinewidth=2,
            mirror=True,
            showline=True,
            linewidth=2,
            linecolor="black",
            title_standoff=20,
            tickmode='linear',  # Ensures even spacing
            dtick=1  # Forces equal spacing between x-axis categories
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.3)",
            gridwidth=1,
            title_font=dict(size=12),
            tickfont=dict(size=12, family="Arial", color="black"),
            zeroline=True,
            zerolinewidth=2,
            range=[y_min - y_range_buffer, y_max + y_range_buffer],
            mirror=True,
            showline=True,
            linewidth=2,
            linecolor="black",
        ),
        title=dict(font_size=16),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-1.5,  # Adjusted for Streamlit display
            xanchor="center",
            x=0.5
        )
    )
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,  # Start before first scenario
        x1=len(df_filtered['Scenario'].unique()) - 0.5,  # End after last scenario
        y0=0,
        y1=0,
        line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
    )
    
    # Display the plot directly in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistical summary
    # st.subheader("Statistical Summary")
    # stats_summary = df_filtered[['Scenario', 'Mean', 'Median', 'Q1', 'Q3', 'Min', 'Max', 'Count']].drop_duplicates()