# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor

from sklearn               import metrics

import time

# %%
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 10})

# %%
# Aquifers
# Aquifers
Auser_path = './datasets/Aquifer_Auser.csv'
Doganella_path = './datasets/Aquifer_Doganella.csv'
Luco_path = './datasets/Aquifer_Luco.csv'
Petrignano_path = './datasets/Aquifer_Petrignano.csv'

# Lake
Bilancino_path = './datasets/Lake_Bilancino.csv'

# River
Arno_path = './datasets/River_Arno.csv'

# Water springs
Amiata_path = './datasets/Water_Spring_Amiata.csv'
Lupa_path = './datasets/Water_Spring_Lupa.csv'
Madonna_path = './datasets/Water_Spring_Madonna_di_Canneto.csv'

# %%
Auser_path

# %%
# Column names for target variables
targets = {
    'Auser': [
        'Depth_to_Groundwater_SAL',
        'Depth_to_Groundwater_CoS',
        'Depth_to_Groundwater_LT2'
        ],
    'Doganella': [
        'Depth_to_Groundwater_Pozzo_1',
        'Depth_to_Groundwater_Pozzo_2',
        'Depth_to_Groundwater_Pozzo_3',
        'Depth_to_Groundwater_Pozzo_4',
        'Depth_to_Groundwater_Pozzo_5',
        'Depth_to_Groundwater_Pozzo_6',
        'Depth_to_Groundwater_Pozzo_7',
        'Depth_to_Groundwater_Pozzo_8',
        'Depth_to_Groundwater_Pozzo_9'
        ],
    'Luco': [
        'Depth_to_Groundwater_Podere_Casetta'
        ],
    'Petrignano': [
        'Depth_to_Groundwater_P24',
        'Depth_to_Groundwater_P25'
        ],
    'Bilancino': [
        'Lake_Level', 
        'Flow_Rate'
        ],
    'Arno': [
        'Hydrometry_Nave_di_Rosano'
        ],
    'Amiata': [
        'Flow_Rate_Bugnano',
        'Flow_Rate_Arbure',
        'Flow_Rate_Ermicciolo',
        'Flow_Rate_Galleria_Alta'
        ],
    'Lupa': [
        'Flow_Rate_Lupa'
        ],
    'Madonna': [
        'Flow_Rate_Madonna_di_Canneto'
        ]
    }

# %%
# Models to be compared
models = {
  "Regression":    LinearRegression(),
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100)
}

# %%
# Splits and shuffle for cross-validation
kf = KFold(3, shuffle=True, random_state=1)

# %%
# For applying various data frequencies
resampling = {'monthly': 'M', 'weekly': 'W', 'daily': 'D'}

# %%
def plot_nans(df: pd.DataFrame, obj_id: str):
    """Function calculates percentage of missing values by column
    and creates a bar plot."""
    rows, _ = df.shape
    missing_values = df.isna().sum() / rows * 100
    missing_values = missing_values[missing_values != 0]
    missing_values.sort_values(inplace=True)
    title = obj_id + ' missing values'
    plt.barh(missing_values.index, missing_values.values)
    plt.xlabel('Percentage (%)')
    plt.title(title)
    plt.show()

# %%
def plot_distribution(df: pd.DataFrame):
    """Function plots a histogram for parameter distribution."""
    df.hist(bins=20, figsize=(14, 10))
    plt.show()

# %%
def plot_correlation(df: pd.DataFrame, obj_id: str, targets: list):
    """Function calculates correlation between parameters
    and creates a heatmap."""
    title = obj_id + ' Heatmap'
    targets_correlation = df.corr()[targets]
    ax = sns.heatmap(targets_correlation, center=0, annot=True, cmap='RdBu_r')
    l, r = ax.get_ylim()
    ax.set_ylim(l + 0.5, r - 0.5)
    plt.yticks(rotation=0)
    plt.title(title)
    plt.show()

# %%
def plot_timeseries(df: pd.DataFrame, obj_id: str, targets: list):
    """Function plots target variable against the timescale."""
    for target in targets:
        plt.plot(df.index, df[target], label=target)
    title = obj_id + ' actual data'
    plt.legend()
    plt.title(title)
    plt.show()

# %%
def plot_seasonality(df: pd.DataFrame, targets: list):
    """Function creates a seasonal decomposition plot for target variables.
    Temporary interpolation of missing values is performed on a resampled
    monthly data, which does not affect the original dataset."""
    for target in targets:
        monthly_interpolated = df[target].resample('M').mean().interpolate(method='akima').dropna()
        
        decomposition = seasonal_decompose(monthly_interpolated)
        observed = decomposition.observed
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        dates = monthly_interpolated.index

        plt.plot(dates, observed, label='Original data')
        plt.plot(dates, trend, label='Trend')
        plt.plot(dates, seasonal, label='Seasonal')
        plt.plot(dates, residual, label='Residual')
        plt.legend()
        plt.title(f'{target} seasonal decomposition')
        plt.tight_layout()
        plt.show()

# %%
def data_cleaning(df: pd.DataFrame):
    """Function replaces 0 values with np.nan in all columns except rainfall."""
    for column in df.columns:
        if column.find('Rainfall') == -1:
            df[column] = df[column].apply(lambda x: np.nan if x == 0 else x)
    return df

# %%
def get_data(path: str):
    """Function extracts data from a csv file and converts date column
    to datetime index."""
    df = pd.read_csv(path,
                       parse_dates=['Date'],
                       date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    df.dropna(subset=['Date'], inplace=True)  # Madonna_di_Canneto dataset has empty rows
    df.set_index('Date', inplace=True)
    # Remove erroneous 0 values from all columns except rainfall
    df = data_cleaning(df)
    return df

# %%
def resample_data(df: pd.DataFrame, freq: str):
    """Function converts daily data into weekly or monthly averages."""
    return df.resample(freq).mean()

# %%
def add_seasonality(df: pd.DataFrame):
    """Function adds columns specifying year, month and day of a year
    and binary column for rainy season (October through April)."""
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Week'] = df.index.weekofyear
    df['Day'] = df.index.dayofyear
    df['Rainy_Season'] = df['Month'].apply(lambda x: 0 if 5 <= x <= 9 else 1)
    return df

# %%
def add_weekly_averages(df: pd.DataFrame):
    """Function adds weekly rolling average values for rainfall and temperature,
    which are used as additional features in daily datasets."""
    for column in df.columns:
        if column.find('Rainfall') > -1 or column.find('Temperature') > -1:
            df[f'{column}_weekly'] = df[column].rolling(7).mean()
    return df

# %%
def select_features(df: pd.DataFrame):
    """Function creates a list of most correlated features for the target."""
    target_correlation = df.corr()['Target']
    mosts_correlated = target_correlation[(target_correlation >= 0.2) | (target_correlation <= -0.2)].index.tolist()
    return mosts_correlated

# %%
def get_X_y(df: pd.DataFrame, target: str, steps_ahead: int):
    """Function splits data into input features and target variable,
    returns a tuple containing list of features, X and y."""
    X = df.copy()
    # Add a column that contains target value for the predicted future period
    # (shift target column backwards for required number of periods)
    X['Target'] = X[target].shift(-steps_ahead)
    # Reduce input to the most correlated features
    features = select_features(X)
    X = X[features]
    # Here the last row of actual inputs is lost
    # because there is no future period target value for it
    X.dropna(inplace=True)
    y = X.pop('Target')
    return features[:-1], X, y

# %%
def evaluate_models(input_data: pd.DataFrame, target: pd.Series, target_name: str):
    """Function estimates cross-val score for several models.
    If the highest cv score is above 0.6, returns a tuple
    with fitted model and its name, otherwise returns a tuple
    with None and an empty string."""

    best_cv = -1
    best_model = None
    best_model_name = ''

    for name, model in models.items():
        cv_r2 = cross_val_score(model, input_data, target, cv=kf, scoring='r2').mean()
        cv_mae = - cross_val_score(model, input_data, target, cv=kf, scoring='neg_mean_absolute_error').mean()
        cv_rmse = - cross_val_score(model, input_data, target, cv=kf, scoring='neg_mean_squared_error').mean()
        print(f'{name} cross-val score for {target_name}:\n\tR2 = {cv_r2}\n\tMAE = {cv_mae}\n\tRMSE = {cv_rmse}')

        if cv_r2 > best_cv:
            best_cv = cv_r2
            best_model = model
            best_model_name = name

    if best_cv >= 0.6:
        best_model.fit(input_data, target)
        return best_model_name, best_model
    else:
        return '', None

# %%
def simple_prediction(ts: pd.Series, n_periods_1: int, n_perionds_2: int, steps_ahead: int):
    """Function returns a prediction based on the actual value of the target variable
    for the latest period and two linear trend predictions equally weighted."""
    # Actual last value in the time series
    last_period_value = ts.iloc[len(ts) - 1]
    # Linear trend based on n_periods_1
    X_1 = np.array([i for i in range(1, n_periods_1 + 1)]).reshape(-1, 1)
    linear_prediction_1 = LinearRegression().fit(X_1, ts.tail(n_periods_1)).predict([[n_periods_1 + steps_ahead]])[0]
    # Linear trend based on n_periods_2
    X_2 = np.array([i for i in range(1, n_perionds_2 + 1)]).reshape(-1, 1)
    linear_prediction_2 = LinearRegression().fit(X_2, ts.tail(n_perionds_2)).predict([[n_perionds_2 + steps_ahead]])[0]
    # Average of the three values
    prediction = (last_period_value + linear_prediction_1 + linear_prediction_2) / 3
    return prediction

# %%
def modelling(df: pd.DataFrame, targets: list, obj_id: str, freq: str, steps_ahead: int):
    """Function preprocesses data, creates models and estimates their accuracy,
    gets prediction for the future period from the best model if R2 >= 0.6
    or uses simple prediction based on the last actual value and linear trends."""
    print(f'\nCreating {freq} model for {obj_id}\n')
    df = resample_data(df, resampling[freq])  # Change data frequency
    if freq == 'daily':
        df = add_weekly_averages(df)  # Add weekly rolling averages as a feature

    # Select input for each target that contains the most correlated features
    for target in targets:
        features, X, y = get_X_y(df, target, steps_ahead)
        model_name, model = evaluate_models(X, y, target)

        if model_name:  # Best R2 >= 0.6
            # Get the actual last row of input data from the dataset
            # (if there are NaNs, get the last row with all required features)
            input_data = df[features].dropna()
            input_date = input_data.index.max()
            input_data = input_data.iloc[len(input_data) - 1, :].values.reshape(1, -1)
            prediction = model.predict(input_data)[0]
        else:  # Low R2
            model_name = 'Average and linear trend'
            features = [target]
            input_data = df[target].dropna()
            input_date = input_data.index.max()
            # Take into account last value and trends of the last 5 and 10 periods
            prediction = simple_prediction(input_data, 5, 10, steps_ahead)

        print(f'\n{model_name} {freq} prediction for {target}: {prediction}')
        print(f'\nInput features: {", ".join(features)}\nInput date: {input_date}')
        print(f'Prediction for {steps_ahead} step(s) ahead.\n')

# %% [markdown]
# ### Aquifer Auser

# %%
data = get_data(Auser_path)

# %%
target_cols = targets['Auser']

# %%
plot_nans(data, 'Auser aquifer')

# %%
plot_distribution(data)

# %%
plot_correlation(data, 'Auser Aquifer', target_cols)

# %%
plot_timeseries(data, 'Auser Aquifer', target_cols)

# %%
plot_seasonality(data, target_cols)

# %%
# Feature engineering
data = add_weekly_averages(data)
data = add_seasonality(data)

# %%
# Create monthly models and make a forecast
modelling(data, target_cols, 'Auser Aquifer', 'monthly', 1)

# %%
# Create daily models and make a forecast
modelling(data, target_cols, 'Auser Aquifer', 'daily', 1)

# %% [markdown]
# ### Lake Bilancino

# %%
data = get_data(Bilancino_path)

# %%
target_cols = targets['Bilancino']

# %%
plot_nans(data, 'Bilancino lake')

# %%
plot_distribution(data)

# %%
plot_correlation(data, 'Bilancino lake', target_cols)

# %%
plot_timeseries(data, 'Bilancino lake', target_cols)

# %%
plot_seasonality(data, target_cols)

# %%
# Feature engineering
data = add_weekly_averages(data)
data = add_seasonality(data)

# %%
# Create monthly models and make a forecast
modelling(data, target_cols, 'Bilancino lake', 'monthly', 1)

# %%
# Create daily models and make a forecast
modelling(data, target_cols, 'Bilancino lake', 'daily', 1)

# %% [markdown]
# ### Waterspring AMIATA

# %%
data = get_data(Amiata_path)

# %%
target_cols = targets['Amiata']

# %%
plot_nans(data, 'Amiata water spring')

# %%
plot_distribution(data)

# %%
plot_correlation(data, 'Amiata water spring', target_cols)

# %%
plot_timeseries(data, 'Amiata water spring', target_cols)

# %%
plot_seasonality(data, target_cols)

# %%
# Feature engineering
data = add_weekly_averages(data)
data = add_seasonality(data)

# %%
# Create monthly models and make a forecast
modelling(data, target_cols, 'Amiata water spring', 'monthly', 1)

# %%
# Create daily models and make a forecast
modelling(data, target_cols, 'Amiata water spring', 'daily', 1)


