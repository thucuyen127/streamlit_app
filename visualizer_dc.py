import streamlit as st
# import sys
import argparse
from visualizer_utils import *
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")


# # Function to handle command-line arguments from sys.argv
# def parse_args():
#     # Check if there are any arguments after the script name
#     if len(sys.argv) > 1:
#         return sys.argv[1]  # Return the first argument after the script name (assumed to be the path)
#     return None


def parse_args():
    parser = argparse.ArgumentParser(description="Process output folder and list of JSON files")

    # Argument for output folder (-o)
    parser.add_argument('-o', '--output_folder', 
                        type=str, 
                        required=False, 
                        help="Path to the output folder", 
                        default='')

    # Argument for a list of JSON files (-l)
    parser.add_argument('-l', '--list_json', 
                        nargs='+', 
                        required=False, 
                        help="List of JSON files", 
                        default=None)

    # Argument for a file containing paths (-f)
    parser.add_argument('-f', '--path_file', 
                        type=str, 
                        required=False, 
                        help="Path to .txt or .json file containing a list of file paths", 
                        default=None)

    # Argument for mode (-mode)
    parser.add_argument('-mode', 
                        type=str, 
                        required=False, 
                        help="Set mode to 'ghg' or 'er'", 
                        default=None)

    # Argument for result CSV (-r)
    parser.add_argument('-r', '--result_csv', 
                        type=str, 
                        required=False, 
                        help="Path to the result CSV file", 
                        default=None)

    # New argument for folder containing JSON files (-j)
    parser.add_argument('-j', '--folder_json_path', 
                        type=str, 
                        required=False, 
                        help="Path to folder containing JSON files", 
                        default=None)

    # Parse the arguments
    args = parser.parse_args()

    # Convert mode argument to a specific value
    if args.mode:
        mode_lower = args.mode.lower()
        if mode_lower in ['ghg', 'true']:
            mode = 'ghg'
        elif mode_lower == 'er':
            mode = 'er'
        else:
            mode = False
    else:
        mode = False

    return args.output_folder, args.list_json, args.path_file, mode, args.result_csv, args.folder_json_path

# Process the command-line arguments
path_output_folder, list_json, path_txt_file, mode, result_csv, folder_json_path = parse_args()



# Ensure the directory exists
Path(path_output_folder).mkdir(parents=True, exist_ok=True)


# Page configuration
st.set_page_config(
    page_title="DAYCENT Model Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS với thiết kế hiện đại
st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Header styles */
        .main-header {
            background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
            padding: 0.3rem;
            border-radius: 3px; 
            margin-bottom: 0.3rem; 
            color: white;
            text-align: center;
            font-size: 1rem; 
        }
            
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
            padding: 2rem;
        }
        
        /* Card-like containers */
        .stBlock {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
        }
        
        /* Input fields */
        .stSelectbox>div>div,
        .stMultiselect>div>div {
            background-color: white;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        
        /* Slider styling */
        .stSlider>div>div {
            background-color: #f0f2f6;
        }
        
        /* File uploader */
        .stFileUploader>div>div {
            border: 2px dashed #1E88E5;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
        }
        
        /* Plot container */
        .plot-container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Loading spinner */
        .stSpinner>div {
            border-top-color: #1E88E5 !important;
        }
        
        /* Alert messages */
        .stAlert {
            border-radius: 10px;
            padding: 1rem;
        }
        
    </style>
""", unsafe_allow_html=True)

def create_header():
    """Create a custom header with gradient background"""
    st.markdown(
        """
        <div class="main-header">
            <h1>🌰 DAYCENT Model Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_sidebar():
    """Create an organized sidebar with collapsible sections"""
    with st.sidebar:
        st.markdown("""
            <div style='background: #f8f9fa;
                    padding: 0.5rem;
                    border-radius: 3px;
                    border-bottom: 2px solid #1E88E5;
                    margin-bottom: 1.2rem;'>
                <p style='color: #0d47a1; 
                        text-align: left; 
                        margin: 0;
                        font-size: 1.1rem;
                        font-weight: 600;
                        letter-spacing: 0.3px;'>
                    ⚡ Control Panel
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Observed Data Section with Expander
        with st.expander("🧐 Observed Data Source", expanded=False):
            observed_files = st.file_uploader(
                "Upload observed data files (CSV/JSON):",
                type=["csv", "json"],
                accept_multiple_files=True,
                key="observed_data_upload",
                help="Upload one or multiple observed data files in CSV or JSON format."
            )

        # Visual separator
        st.markdown("<hr style='margin: 0.3rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

        # Modeled Data Section with Expander
        with st.expander("📊 Modeled Data Source", expanded=False):
            upload_mode = st.radio(
                "Choose your data source:",
                ("📁 Upload JSON Files", "📂 Select from Output Folder"),
                key="modeled_data_source",
                help="Select how you want to input your modeled data"
            )

    return upload_mode, observed_files


def handle_file_upload(preloaded_files):
    """
    Ultra-compact UI for managing JSON file uploads and preloaded files
    with minimal spacing and native Streamlit remove buttons.
    
    Parameters:
        preloaded_files (list): A list of preloaded JSON files or file names.
        
    Returns:
        list: Combined list of remaining preloaded files and newly uploaded files.
    """
    with st.expander("📂json files", expanded=False):
        # Custom CSS with modern design
        st.markdown(
            """
            <style>
                /* Compact file upload area */
                .css-1upf8sy {
                    padding: 0;
                    margin-bottom: 10px;
                }
                /* Reduce spacing between file items */
                .row-widget.stButton {
                    margin-bottom: 0;
                    padding: 0;
                }
                /* Styling for file names */
                .file-item {
                    padding: 5px;
                    margin: 3px 0;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    background-color: #f0f2f6;
                }
                /* Remove button styling */
                .remove-btn {
                    color: #ff4b4b;
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-weight: bold;
                }
                /* Overall container padding */
                .file-container {
                    padding: 5px 0;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        uploaded_files = st.file_uploader(
            ".", type="json", accept_multiple_files=True, label_visibility="collapsed"
        ) or []
        
        preloaded_files = preloaded_files or []
        # Initialize session state for tracking files
        if "active_files" not in st.session_state:
            st.session_state["active_files"] = preloaded_files.copy()
            
        # Files to be removed in this session
        files_to_remove = []
        
        # Only show the list if there are files
        st.markdown('<div class="file-container">', unsafe_allow_html=True)
        if st.session_state["active_files"]:
            # Create a minimal file list
            for idx, file in enumerate(st.session_state["active_files"]):
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    st.markdown(f'<div class="file-item">{file}</div>', unsafe_allow_html=True)
                with cols[1]:
                    if st.button("✕", key=f"remove_{idx}", help=f"Remove {file}"):
                        files_to_remove.append(file)
            
            # Process removals after rendering to avoid UI issues
            for file in files_to_remove:
                st.session_state["active_files"].remove(file)
                st.rerun()  # Use st.rerun() instead of experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Return combined list of preloaded and uploaded files
    return st.session_state["active_files"] + uploaded_files




# def create_scenario_filters(table):
#     """Create hierarchical filters for scenarios with simplified location selection"""
#     st.sidebar.markdown("### 📍 Location Selection")
    
#     # Region selection
#     region = st.sidebar.selectbox(
#         '🌐 Select Region',
#         options=sorted(table['region'].unique()),
#         help="Select a geographical region"
#     )
#     filtered_by_region = table[table['region'] == region]
    
#     # Province selection
#     province = st.sidebar.selectbox(
#         '🏠 Select Province',
#         options=sorted(filtered_by_region['province'].unique()),
#         help="Select a province within the region"
#     )
#     filtered_by_province = filtered_by_region[filtered_by_region['province'] == province]

#     # Option 1: Chỉ dùng province
#     scenarios = filtered_by_province.index.tolist()
    
#     # Scenario selection
#     selected_scenarios = st.sidebar.multiselect(
#         '📊 Select Scenarios to Compare',
#         options=scenarios,
#         default=scenarios[:2] if len(scenarios) > 1 else scenarios,
#         help="Select multiple scenarios to compare"
#     )
    
#     return selected_scenarios


def main(uploaded_files=None):
    
    # Create header
    create_header()

    # Create sidebar and get upload mode
    upload_mode, observed_files = create_sidebar()

    # Initialize obs_data as an empty list
    obs_data = []

    # Process uploaded files
    if observed_files:
        st.subheader("Uploaded Files")
        file_names = [file.name for file in observed_files]

        for file in observed_files:
            try:
                # Remove file extension (.csv or .json)
                file_base_name = os.path.splitext(file.name)[0]

                if file.name.endswith(".csv"):
                    data = pd.read_csv(file, parse_dates=["Date"], dayfirst=True)
                    # Process time column
                    data = process_time_column(data)
                    
                    obs_data.append(data)  # Append processed data to obs_data

                elif file.name.endswith(".json"):
                    data = process_json_without_result(file)
                    if isinstance(data, pd.DataFrame):
                        # Ensure JSON data is a DataFrame and add "Scenario" column
                        data["Scenario"] = f"observed_{file_base_name}"
                        data = process_time_column(data)
                    else:
                        # If JSON is not a DataFrame, convert it to one
                        data = pd.DataFrame([{"Scenario": f"observed_{file_base_name}", **data}])
                    
                    obs_data.append(data)

            except Exception as e:
                st.error(f"Failed to read file {file.name}: {e}")

    # Merge all DataFrames in obs_data
    try:
        if obs_data:
            # Filter valid DataFrames
            valid_dfs = [df for df in obs_data if isinstance(df, pd.DataFrame)]
            # Merge all valid DataFrames
            obs_data_merge = valid_dfs[0] if len(valid_dfs) == 1 else pd.concat(valid_dfs, ignore_index=True) if valid_dfs else None
        else:
            obs_data_merge = None
    except Exception as e:
        st.error(f"Error merging observed data: {e}")
        obs_data_merge = None


    # Handle different data sources
    if upload_mode == "📁 Upload JSON Files":
        uploaded_files_final = handle_file_upload(uploaded_files)
        
        mode_options = ["None", "GHG Emissions", "Emission Reduction"]
        mode_index = 0  # Default "None"
        
        if mode == "ghg":
            mode_index = 1  # GHG Emissions
        elif mode == "er":
            mode_index = 2  # Emission Reduction

        # Sidebar option for mode selection
        mode_selection = st.sidebar.radio(
            "Select Processing Mode:",
            options=mode_options,
            index=mode_index,  
            help="Choose between GHG emission calculation and Emission Reduction analysis."
        )

        # Determine the selected mode
        calculate_results = mode_selection == mode_options[1]  # "GHG Emissions"
        calculate_er = mode_selection == mode_options[2]
        

        # if uploaded_files_final:
        #     with st.spinner("Processing uploaded files..."):
        try:
            if mode_selection == "GHG Emissions":
                # calc GHG for each json file
                dataframes = []
                for file in uploaded_files_final:
                    df_merge = process_json_with_result(file)
                    if df_merge is not None:
                        df, errors = calc_GHG(df_merge)
                        if isinstance(file, str):
                            file = file.replace("\\", "/")
                            scen = file.split("/")[-1].replace('.json', '')
                        else:
                            scen = ".".join(file.name.split(".")[:-1])

                        if df is not None:
                            df['Scenario'] = scen
                            dataframes.append(df)

                        if errors:
                            if isinstance(errors, list):
                                # Combine all missing variables into a single warning
                                missing_vars = "\n".join([f"- {var}" for var in errors])
                                st.warning(f"Missing variables in scenario '{scen}':\n{missing_vars}")
                            else:
                                st.error(f"Processing error in scenario '{scen}': {errors}")
                if dataframes:
                    plot_scenarios_streamlit(dataframes, obs_data_merge)
            else:
                if mode_selection == "Emission Reduction":
                    mode_upload = st.radio("Select Mode", ["Upload JSON", "Check & Load Result CSV"])

                    if mode_upload == "Upload JSON":
                        with st.expander("📋 JSON File Upload Guide"):
                            st.markdown("""
                            ### JSON File Naming Convention
                            
                            For proper scenario identification, name your JSON files using the following format:
                            
                            ```
                            latitude_longitude_scenarioname_output.json
                            ```
                            
                            #### Example:
                            - ✅ Valid: `23.092409_69.221603_gn_RI_furr_nocc_output.json`
                            - ✅ Valid: `23.092409_69.221603_gn_RI_furr_nocc.json`
                            """)

                        # Optional baseline content input
                        baseline_content = st.text_input("Enter Baseline Content (optional):").strip()
                        # Convert the input string to a list
                        if baseline_content:
                            baseline_content = [item.strip() for item in baseline_content.split(',')]
                        else:
                            baseline_content = None
                        
                        # Displaying a label for factors input
                        
                        factor = st.text_input(
    "Please enter the factors (comma-separated) for analysis, Example: ct, ct-m, gn, gn-m:",
                            value="ct, ct-m, gn, gn-m"  # Default value
                        ).strip()

                        # Convert the input string to a list
                        if factor:
                            factor = [item.strip() for item in factor.split(',')]
                        else:
                            factor = None

                        dataframes = []  
                        for file in uploaded_files_final:  
                            df_merge = process_json_with_result(file)  
                            if df_merge is not None:  
                                df, errors = calc_GHG(df_merge)  

                                # Extract filename
                                if isinstance(file, str):  
                                    file = file.replace("\\", "/")  
                                    filename = file.split("/")[-1].replace('.json', '')  
                                else:  
                                    filename = ".".join(file.name.split(".")[:-1])  

                                # Extract scenario name from filename
                                parts = filename.split('_')
                                if len(parts) >= 3:
                                    try:
                                        lat, lon = float(parts[0]), float(parts[1])  # Extract latitude and longitude
                                        scenario_parts = parts[2:]  

                                        # Remove "output" suffix if present
                                        if scenario_parts and scenario_parts[-1].lower() == "output":
                                            scenario_parts = scenario_parts[:-1]

                                        scenario = "_".join(scenario_parts)  # Construct scenario name
                                    except ValueError:
                                        scenario = filename  # Fallback if lat/lon cannot be converted
                                else:
                                    scenario = filename  # Fallback for unexpected filename format

                                if df is not None:  
                                    df['Scenario'] = scenario  
                                    df['location'] = f'{lat}_{lon}'
                                    df_result = calculate_mean_ghg_by_scenario(df)
                                    df_result['Scenario'] = scenario  
                                    df_result['location'] = f'{lat}_{lon}'
                                    dataframes.append(df_result)  

                                # Handle errors
                                if errors:  
                                    if isinstance(errors, list):  
                                        missing_vars = "\n".join([f"- {var}" for var in errors])  
                                        st.warning(f"Missing variables in scenario '{scenario}':\n{missing_vars}")  
                                    else:  
                                        st.error(f"Processing error in scenario '{scenario}': {errors}")

                        if dataframes:
                            # Concatenate all non-empty DataFrames into a single DataFrame
                            df_result = pd.concat(dataframes, ignore_index=True)

                            box_plot(df_result, baseline_content=baseline_content)
                            # Create an instance of BaselineProcessor
                            processor = BaselineProcessor(df_result)
                            # Retrieve numeric columns
                            numeric_columns = df_result.select_dtypes(include=['float64', 'int64']).columns.tolist()

                            # Allow users to select only one column for boxplots
                            selected_column = st.selectbox(
                                "Select a column to plot",  # Single selection
                                options=numeric_columns,
                                index=0 if numeric_columns else None  # Default to the first column if there are any
                            )
                            # Calculate the DataFra4me with relative differences
                            df_dif = processor.relative_different(selected_column=selected_column, base_content=baseline_content)
                            plot_relative_difference(df_dif,baseline_content=baseline_content, factor_list=factor)
                        else:
                                st.error("No data available for plotting.")
                            

                    elif mode_upload == "Check & Load Result CSV":

                        # File uploader for CSV
                        uploaded_csv = st.file_uploader("Upload Result CSV File", type=['csv'])

                        # Check if the specified CSV path exists before reading
                        df_result_path = None

                        if result_csv:
                            if os.path.exists(result_csv):
                                st.success("CSV file found! Loading data...")
                                df_result_path = pd.read_csv(result_csv)
                            else:
                                st.warning("CSV file path provided, but file does not exist.")
                    
                        # Process uploaded CSV file
                        df_result_uploaded = None
                        if uploaded_csv:
                            uploaded_csv.seek(0)  # Reset file pointer before reading
                            df_result_uploaded = pd.read_csv(uploaded_csv)
                            st.success("CSV uploaded and loaded successfully!")

                        # Let the user select between existing and uploaded CSV if both are available
                        if df_result_path is not None and df_result_uploaded is not None:
                            selected_source = st.radio("Select Data Source:", ["Use Existing CSV", "Use Uploaded CSV"])
                            df_result = df_result_path if selected_source == "Use Existing CSV" else df_result_uploaded
                        elif df_result_path is not None:
                            df_result = df_result_path
                        elif df_result_uploaded is not None:
                            df_result = df_result_uploaded
                        else:
                            df_result = None

                        # Optional baseline content input
                        baseline_content = st.text_input("Enter Baseline Content (optional):", value='noRI_furr_wR0_nocc, noRI_furr_wR90_nocc').strip()
                        # Convert the input string to a list
                        if baseline_content:
                            baseline_content = [item.strip() for item in baseline_content.split(',')]
                        else:
                            baseline_content = None
                        
                        # Displaying a label for factors input
                        
                        factor = st.text_input(
    "Please enter the factors (comma-separated) for analysis, Example: ct, ct-m, gn, gn-m:",
                            value="ct, ct-m, gn, gn-m"  # Default value
                        ).strip()

                        # Convert the input string to a list
                        if factor:
                            factor = [item.strip() for item in factor.split(',')]
                        else:
                            factor = None

                        # Display the selected CSV data
                        if df_result is not None:
                            # Create an instance of BaselineProcessor
                            processor = BaselineProcessor(df_result)
                            # Retrieve numeric columns
                            numeric_columns = df_result.select_dtypes(include=['float64', 'int64']).columns.tolist()

                            # Allow users to select only one column for boxplots
                            selected_column = st.selectbox(
                                "Select a column to plot",  # Single selection
                                options=numeric_columns,
                                index=0 if numeric_columns else None  # Default to the first column if there are any
                            )
                            # Calculate the DataFra4me with relative differences
                            df_dif = processor.relative_different(selected_column=selected_column, base_content=baseline_content)
                            plot_relative_difference(df_dif,baseline_content=baseline_content, factor_list=factor)

                        
                else:
                    if mode_selection == mode_options[0]:
                        all_daily_data = []  # List to store (list_daily_dfs, daily_keys) for each file
                        dataframes_ao = []
                        dataframes_harvest = []
                        
                        for file in uploaded_files_final:
                            daily_data, df_h, df_ao = process_json_without_result(file)
                            
                            # Process daily data (tuple of list_daily_dfs and daily_keys)
                            if daily_data is not None and len(daily_data) == 2:
                                list_daily_dfs, daily_keys = daily_data
                                
                                # Check if list_daily_dfs is not empty
                                if list_daily_dfs:
                                    # Add scenario name to each dataframe in the list
                                    processed_daily_dfs = []
                                    for df in list_daily_dfs:
                                        df_copy = df.copy()
                                        if isinstance(file, str):
                                            file_path = file.replace("\\", "/")
                                            df_copy['Scenario'] = file_path.split("/")[-1].replace('.json', '')
                                        else:
                                            df_copy['Scenario'] = ".".join(file.name.split(".")[:-1])
                                        processed_daily_dfs.append(df_copy)
                                    
                                    all_daily_data.append((processed_daily_dfs, daily_keys))
                            
                            if df_h is not None:
                                if isinstance(file, str):
                                    file = file.replace("\\", "/")
                                    df_h['Scenario'] = file.split("/")[-1].replace('.json', '')
                                else:
                                    df_h['Scenario'] = ".".join(file.name.split(".")[:-1])
                                dataframes_harvest.append(df_h)
                            
                            if df_ao is not None:
                                if isinstance(file, str):
                                    file = file.replace("\\", "/")
                                    df_ao['Scenario'] = file.split("/")[-1].replace('.json', '')
                                else:
                                    df_ao['Scenario'] = ".".join(file.name.split(".")[:-1])
                                dataframes_ao.append(df_ao)
                        
                        # Use DuckDB version instead of pandas version
                        plot_scenarios_streamlit_duckdb(all_daily_data, dataframes_ao, dataframes_harvest, obs_data_merge)
                
        except Exception as e:
            st.warning(f"Please switch basic method")
            st.warning(f"error: {e}")
        
        
            
    else:  # Select from outputs folder
        try:
            # Display and allow users to edit output path
            path_output = st.text_input("Output Path", value=path_output_folder)

            # Create directory if value changes
            if not os.path.exists(path_output):
                Path(path_output).mkdir(parents=True, exist_ok=True)

            if path_output:
                # Display subfolders in the sidebar for multi-selection
                subfolders = [f.name for f in Path(path_output).iterdir() if f.is_dir()]
                selected_folders = st.sidebar.multiselect("Select folders to process:", subfolders)
                
                # Add calculation checkbox similar to JSON upload mode
                calculate_results = st.sidebar.checkbox(
                                    "Calculate GHG emissions", 
                                    value=False,  # Set to unchecked by default
                                    help="Check this box if you want to calculate GHG emissions."
                                )

                if selected_folders:
                    if calculate_results:
                        # Process with GHG, SOM, YIELD calculations
                        dataframes = []
                        for selected_folder in selected_folders:
                            full_path = Path(path_output) / selected_folder
                            df_merge = process_csv_with_result(full_path)  # You'll need to create this function
                            if df_merge is not None:
                                df, errors = calc_GHG(df_merge, True)


                                if df is not None:
                                    if 'Scenario' not in df.columns:
                                        df['Scenario'] = selected_folder
                                    dataframes.append(df)

                                if errors:
                                    if isinstance(errors, list):
                                        # Combine all missing variables into a single warning
                                        missing_vars = "\n".join([f"- {var}" for var in errors])
                                        st.warning(f"Missing variables in scenario '{selected_folder}':\n{missing_vars}")
                                    else:
                                        st.error(f"Processing error in scenario '{selected_folder}': {errors}")
                                    
                            
                        if dataframes:
                            
                            plot_scenarios_streamlit(dataframes, obs_data_merge)
                    else:
                        # Original processing without calculations
                        dataframes_w = []
                        dataframes_dv = []

                        for selected_folder in selected_folders:
                            full_path = Path(path_output) / selected_folder
                            df_wlis, df_dv = process_csv_without_result(full_path)
                            if df_wlis is not None:
                                if 'Scenario' not in df_wlis.columns:
                                    df_wlis['Scenario'] = selected_folder
                                dataframes_w.append(df_wlis)

                            if df_dv is not None:
                                if 'Scenario' not in df_dv.columns:
                                    df_dv['Scenario'] = selected_folder
                                dataframes_dv.append(df_dv)

                        plot_scenarios_streamlit_csv(dataframes_w, dataframes_dv, obs_data_merge)

    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    
        # Add the refresh button in the sidebar
    if st.sidebar.button('Refresh Visualizer'):
        refresh_visualizer()


    # # Add download button
    # if st.sidebar.button('📥 Download Analysis Results'):
    #     try:
    #         if 'filtered_df' in locals():
    #             st.sidebar.success('✅ Results saved as analysis_results.csv')
    #     except Exception as e:
    #         st.sidebar.error(f"Error saving results: {str(e)}")
    # Footer with team name
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <div style='color: #666; font-size: 0.8rem; margin-bottom: 0.3rem;'>
                DAYCENT Model Analysis Tool
            </div>
            <div style='color: #1E88E5; font-size: 0.75rem; font-weight: 500;'>
                by RegenAI-Solutions
            </div>
        </div>
    """, unsafe_allow_html=True)

def find_streamlit_ports(target_ports={8501, 8502, 8503, 8504, 8505}):
    """
    Check for running Streamlit processes and return a dictionary mapping process PID
    to a set of target local ports used.
    """
    streamlit_ports = {}
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and 'streamlit' in ' '.join(proc.info['cmdline']).lower():
                connections = proc.net_connections(kind='inet')
                ports = {conn.laddr.port for conn in connections 
                         if conn.laddr and conn.laddr.port in target_ports}
                if ports:
                    streamlit_ports[proc.pid] = ports
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return streamlit_ports

def current_streamlit_ports():
    """
    Get the current Streamlit port from configuration.
    """
    return st.config.get_option("server.port")

def kill_other_streamlit_processes():
    """
    Kill all Streamlit processes except the one running on the current port.
    """
    streamlit_ports = find_streamlit_ports()  # Get all running Streamlit ports
    current_port = current_streamlit_ports()  # Get the current Streamlit port
    
    if current_port is None:
        print("⚠️ Current Streamlit port not found!")
        return
    
    # Iterate through each process, and kill those not using the current port
    for pid, ports in streamlit_ports.items():
        if current_port not in ports:
            print(f"🛑 Killing Streamlit process {pid} running on ports {ports}")
            try:
                os.kill(pid, signal.SIGTERM)  # Use SIGTERM to terminate the process
            except PermissionError:
                print(f"⚠️ Unable to kill process {pid} - administrative privileges required.")
            except Exception as e:
                print(f"❌ Error killing process {pid}: {e}")


if __name__ == "__main__":
    kill_other_streamlit_processes()
    jsons = None
    jsons_2 = None

    # Load JSON paths from text/json file
    if path_txt_file:
        jsons = load_paths_from_file(path_txt_file)

    # Load JSON paths from folder
    if folder_json_path:
        jsons_2 = load_paths_from_folder(folder_json_path)

    # Combine all sources of JSON paths
    combined_list = []

    if jsons:
        combined_list.extend(jsons)

    if list_json:
        combined_list.extend(list_json)

    if jsons_2:
        combined_list.extend(jsons_2)

    # If combined_list is still empty, set it to None
    if not combined_list:
        combined_list = None

    # Run main function
    main(combined_list)