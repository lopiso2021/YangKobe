# YangKobe
High resolution gridded daily soil temperature dataset in Ethiopia
######################################################### code for preciction of soil Temp.#################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================ 
# 4 BOOSTING MODELS
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with optimized parameters"""
    try:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model
    except Exception as e:
        return None, None

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with optimized parameters"""
    try:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model
    except Exception as e:
        return None, None

def train_catboost(X_train, y_train, X_test, y_test):
    """Train CatBoost with optimized parameters"""
    try:
        model = CatBoostRegressor(
            iterations=200,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model
    except Exception as e:
        return None, None

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with optimized parameters"""
    try:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model
    except Exception as e:
        return None, None

# ============================================================================ 
# SCENARIOS
# ============================================================================

scenarios = {
    'MEAN-MAX-MIN': ['AMEAN', 'AMAX', 'AMIN'],
    'MEAN-MAX-MIN-RAIN': ['AMEAN', 'AMAX', 'AMIN', 'ARAIN'],
    'MEAN-MAX-MIN-HUM': ['AMEAN', 'AMAX', 'AMIN', 'AHUM'],
    'MEAN-MAX-MIN-HUM-RAIN': ['AMEAN', 'AMAX', 'AMIN', 'AHUM', 'ARAIN']
}

# All 4 boosting models
models_to_train = [
    ('XGBoost', train_xgboost),
    ('LightGBM', train_lightgbm),
    ('CatBoost', train_catboost),
    ('GradientBoosting', train_gradient_boosting)
]

# ============================================================================ 
# DATA PREPARATION FUNCTIONS
# ============================================================================

def simple_clean_data(df):
    """Simple data cleaning"""
    df_clean = df.copy()
    
    # Handle missing values
    missing_indicators = [' -   ', '--', 'NaN', 'N/A', '', 'NA', 'null', 'NULL']
    for indicator in missing_indicators:
        df_clean.replace(indicator, np.nan, inplace=True)
    
    # Convert to numeric
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Fill missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if 'RAIN' in col.upper():
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(df_clean[col].median())
    
    return df_clean

def create_simple_features(df, scenario_features, target_columns):
    """Create features without causing NaN issues"""
    df_temp = df.copy()
    focus_features = scenario_features.copy()
    
    # Add lag features
    for target in target_columns:
        if target in df_temp.columns:
            df_temp[f'{target}_lag1'] = df_temp[target].shift(1)
            focus_features.append(f'{target}_lag1')
    
    # Fill NaN in lag features
    for feature in focus_features:
        if feature in df_temp.columns and df_temp[feature].isnull().any():
            if 'lag' in feature:
                original_target = feature.replace('_lag1', '')
                if original_target in df_temp.columns:
                    df_temp[feature] = df_temp[feature].fillna(df_temp[original_target])
                else:
                    df_temp[feature] = df_temp[feature].fillna(df_temp[feature].median())
            elif 'RAIN' in feature.upper():
                df_temp[feature] = df_temp[feature].fillna(0)
            else:
                df_temp[feature] = df_temp[feature].fillna(method='ffill').fillna(df_temp[feature].median())
    
    # Remove rows where targets are NaN
    target_complete_mask = df_temp[target_columns].notna().all(axis=1)
    df_final = df_temp[target_complete_mask].copy()
    
    return df_final, focus_features

# ============================================================================ 
# STATION DEFINITIONS (CORRECTED)
# ============================================================================

stations = {
    'AB1': 'Arba Minch',
    'AD1': 'Addis Ababa', 
    'AW1': 'Awassa',
    'DW1': 'Dire Dawa',  # CORRECTED: DW1 = Dire Dawa
    'GN1': 'Gondar',
    'HO1': 'Hosana',
    'JJ1': 'Jijiga',
    'JM1': 'Jimma',
    'KN1': 'Konso',
    'NG1': 'Neghele',
    'DM1': 'Debre Markos'  # CORRECTED: DM1 = Debre Markos
}

# Define all file paths
file_paths = {
    'AB1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\AB1.csv',
    'AD1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\AD1.csv',
    'AW1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\AW1.csv',
    'DW1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\DW1.csv',  # Dire Dawa
    'GN1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\GN1.csv',
    'HO1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\HO1.csv',
    'JJ1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\JJ1.csv',
    'JM1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\JM1.csv',
    'KN1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\KN1.csv',
    'NG1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\NG1.csv',
    'DM1': r'C:\Users\fekadu\OneDrive\Desktop\DataARt\soil_Tmap_observed\1Day_TW\PR_TW1\DM1.csv'   # Debre Markos
}

# ============================================================================ 
# MAIN PROCESSING - ALL STATIONS
# ============================================================================

print(f"4 BOOSTING MODELS - 4 SCENARIOS - 11 STATIONS")
print(f"Station Corrections: DW1 = Dire Dawa, DM1 = Debre Markos")

# Store all results
all_station_results = []

for station_code, file_path in file_paths.items():
    station_name = stations[station_code]
    print(f"PROCESSING: {station_name} ({station_code})")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Clean data
        df_clean = simple_clean_data(df)
        
        # Define targets
        target_00hr = "ST20-00"
        target_12hr = "ST20-12"
        available_targets = [target_00hr, target_12hr]
        
        # Process each scenario
        for scenario_name, scenario_features in scenarios.items():
            # Check if all base features exist
            missing_features = [f for f in scenario_features if f not in df_clean.columns]
            if missing_features:
                continue
            
            # Create features
            df_enhanced, all_features = create_simple_features(df_clean, scenario_features, available_targets)
            features = [f for f in all_features if f not in available_targets]
            
            # Process each model
            for model_name, train_function in models_to_train:
                rmse_00hr_values = []
                rmse_12hr_values = []
                r_00hr_values = []
                r_12hr_values = []
                
                # Train for both targets
                for target_column in available_targets:
                    if target_column not in df_enhanced.columns:
                        continue
                        
                    X = df_enhanced[features].copy()
                    y = df_enhanced[target_column].copy()
                    
                    # Final check - remove any remaining NaN
                    mask = X.notna().all(axis=1) & y.notna()
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) < 10:
                        continue
                    
                    # Convert to arrays
                    X_array = X_clean.values
                    y_array = y_clean.values
                    
                    # Time-based split
                    split_point = int(len(X_array) * 0.8)
                    X_train = X_array[:split_point]
                    X_test = X_array[split_point:]
                    y_train = y_array[:split_point]
                    y_test = y_array[split_point:]
                    
                    # Train model
                    y_pred, model = train_function(X_train, y_train, X_test, y_test)
                    
                    if y_pred is not None and len(y_pred) == len(y_test):
                        # Calculate RMSE value
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        # Calculate R value
                        correlation_matrix = np.corrcoef(y_test, y_pred)
                        r_value = correlation_matrix[0, 1]
                        
                        if target_column == 'ST20-00':
                            rmse_00hr_values.append(rmse)
                            r_00hr_values.append(r_value)
                        else:
                            rmse_12hr_values.append(rmse)
                            r_12hr_values.append(r_value)
                
                # Store results
                if rmse_00hr_values and rmse_12hr_values:
                    avg_rmse_00hr = np.mean(rmse_00hr_values)
                    avg_rmse_12hr = np.mean(rmse_12hr_values)
                    avg_r_00hr = np.mean(r_00hr_values)
                    avg_r_12hr = np.mean(r_12hr_values)
                    
                    all_station_results.append({
                        'Station': station_name,
                        'Station_Code': station_code,
                        'Model': model_name,
                        'Scenario': scenario_name,
                        'RMSE_00hr': round(avg_rmse_00hr, 4),
                        'RMSE_12hr': round(avg_rmse_12hr, 4),
                        'R_00hr': round(avg_r_00hr, 4),
                        'R_12hr': round(avg_r_12hr, 4)
                    })
        
        print(f"COMPLETED: {station_name} - {len([x for x in all_station_results if x['Station_Code'] == station_code])} combinations")
            
    except Exception as e:
        print(f"ERROR: {station_name} - {str(e)}")
        continue

# ============================================================================ 
# CREATE CSV OUTPUTS
# ============================================================================

if all_station_results:
    # Create main results dataframe
    final_df = pd.DataFrame(all_station_results)
    
    # 1. Save all individual results
    final_df.to_csv('All_Stations_Individual_ResultsST20.csv', index=False)
    print(f"\n1. All_Stations_Individual_Results.csv saved - {len(final_df)} rows")
    
    # 2. Calculate and save 16-row overall averages
    overall_16row = final_df.groupby(['Model', 'Scenario']).agg({
        'R_00hr': 'mean',
        'R_12hr': 'mean',
        'RMSE_00hr': 'mean',
        'RMSE_12hr': 'mean'
    }).round(4).reset_index()
    
    overall_16row.to_csv('Overall_16Row_AveragesST20.csv', index=False)
    print(f"2. Overall_16Row_Averages.csv saved - {len(overall_16row)} rows")
    
    # 3. Save station summary
    station_summary = final_df.groupby(['Station', 'Station_Code']).agg({
        'R_00hr': 'mean',
        'R_12hr': 'mean',
        'RMSE_00hr': 'mean',
        'RMSE_12hr': 'mean'
    }).round(4).reset_index()
    
    station_summary.to_csv('Station_SummaryST20.csv', index=False)
    print(f"3. Station_Summary.csv saved - {len(station_summary)} rows")
    
    # Display summary
    print(f"\nSUMMARY:")
    print(f"Total stations processed: {len(station_summary)}")
    print(f"Total model-scenario combinations: {len(final_df)}")
    print(f"Overall 16-row combinations: {len(overall_16row)}")
    
    # Show best overall combination
    overall_16row['Overall_Score'] = (overall_16row['R_00hr'] + overall_16row['R_12hr']) / 2
    best_overall = overall_16row.loc[overall_16row['Overall_Score'].idxmax()]
    
    print(f"\nBEST OVERALL COMBINATION:")
    print(f"Model: {best_overall['Model']}, Scenario: {best_overall['Scenario']}")
    print(f"R_00hr: {best_overall['R_00hr']}, R_12hr: {best_overall['R_12hr']}")
    
else:
    print("No results generated")

print("PROCESSING COMPLETED")
