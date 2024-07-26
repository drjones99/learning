import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    return pd.read_csv(file_path, encoding='latin-1')

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
'''
def cross_validate_model(df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        X_train, y_train, _ = prepare_features(train_data)
        X_test, y_test, _ = prepare_features(test_data)

        model = train_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return {
        'MSE': np.mean(mse_scores),
        'MAE': np.mean(mae_scores),
        'R2': np.mean(r2_scores)
    }
'''
def prepare_features(data, is_future=False):
    df = data.copy()
    
    if not is_future:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        df['day_of_week'] = df['kickoff_time'].dt.dayofweek
        df['month'] = df['kickoff_time'].dt.month
    else:
        df['day_of_week'] = 5  # Assuming Saturday for future games
        df['month'] = 8  # Assuming August for the start of the season
    
    df = df.sort_values(['name', 'GW'])
    
    # Create 'prev_minutes', 'prev_starts', 'rolling_avg_minutes', 'rolling_avg_starts' columns
    if 'minutes' in df.columns:
        df['prev_minutes'] = df.groupby('name')['minutes'].shift(1).fillna(0)
        df['rolling_avg_minutes'] = df.groupby('name')['minutes'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
    else:
        df['minutes'] = 0
        df['prev_minutes'] = 0
        df['rolling_avg_minutes'] = 0
    
    if 'starts' in df.columns:
        df['prev_starts'] = df.groupby('name')['starts'].shift(1).fillna(0)
        df['rolling_avg_starts'] = df.groupby('name')['starts'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
    else:
        df['starts'] = 0
        df['prev_starts'] = 0
        df['rolling_avg_starts'] = 0
    
    le = LabelEncoder()
    df['position_encoded'] = le.fit_transform(df['position'])
    df['team_encoded'] = le.fit_transform(df['team'])
    
    features = ['GW', 'position_encoded', 'team_encoded', 'day_of_week', 'month', 
                'prev_minutes', 'prev_starts', 'rolling_avg_minutes', 'rolling_avg_starts']
    
    return df[features], df['minutes'], df[['name', 'position', 'team', 'GW']]

def create_future_gameweeks(df, num_gameweeks=5):
    last_gw = df['GW'].max()
    players = df[df['GW'] == last_gw].copy()
    
    future_gws = []
    for i in range(1, num_gameweeks + 1):
        gw_data = players.copy()
        gw_data['GW'] = last_gw + i
        gw_data['day_of_week'] = 5  # Assuming Saturday for future games
        gw_data['month'] = 8  # Assuming August for the start of the season
        future_gws.append(gw_data)
    
    future_gws = pd.concat(future_gws, ignore_index=True)
    return future_gws

def calculate_expected_minutes(data, future_data, model):
    X, _, context = prepare_features(data)
    
    all_predictions = []
    for gw in future_data['GW'].unique():
        gw_data = future_data[future_data['GW'] == gw].copy()
        X_future, _, context_future = prepare_features(gw_data, is_future=True)
        
        expected_minutes = model.predict(X_future)
        
        result = context_future.copy()
        result['expected_minutes'] = np.clip(expected_minutes, 0, 90)  # Clip predictions between 0 and 90
        all_predictions.append(result)
        
        # Update future_data for the next gameweek
        future_data.loc[
            (future_data['GW'] > gw) & (future_data['name'].isin(result['name'])),
            'prev_minutes'
        ] = result['expected_minutes'].values

        future_data.loc[
            (future_data['GW'] > gw) & (future_data['name'].isin(result['name'])),
            'prev_starts'
        ] = (result['expected_minutes'] > 60).astype(int).values

        # Update rolling averages
        future_data.loc[future_data['GW'] > gw, 'rolling_avg_minutes'] = (
            future_data[future_data['GW'] > gw].groupby('name')['prev_minutes']
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )

        future_data.loc[future_data['GW'] > gw, 'rolling_avg_starts'] = (
            future_data[future_data['GW'] > gw].groupby('name')['prev_starts']
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
    
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    
    print("\nStatistics of predicted minutes:")
    print(all_predictions['expected_minutes'].describe())
    
    return all_predictions

# Main execution
csv_file = 'test.csv'
df = read_csv_with_encoding(csv_file)

print("Columns in the original DataFrame:")
print(df.columns)

players_with_minutes = df.groupby('name')['minutes'].sum() > 0
active_players = players_with_minutes[players_with_minutes].index
df_filtered = df[df['name'].isin(active_players)].copy()

print(f"\nTotal players: {df['name'].nunique()}")
print(f"Active players: {len(active_players)}")
print(f"Removed {df['name'].nunique() - len(active_players)} players with 0 minutes in all gameweeks")

X, y, _ = prepare_features(df_filtered)

print("\nFeature values for training data:")
print(X.describe())

print("\nTarget variable (minutes) statistics:")
print(y.describe())

final_model = train_random_forest(X, y)

# Create future gameweeks data
future_gws = create_future_gameweeks(df_filtered, num_gameweeks=5)

# Predict minutes for future gameweeks
final_predictions = calculate_expected_minutes(df_filtered, future_gws, final_model)
final_predictions.to_csv('predicted_minutes_next_season.csv', index=False)
print("\nPredicted minutes for the first 5 gameweeks of the new season have been calculated and exported to 'predicted_minutes_next_season.csv'")

# Display sample predictions
print("\nSample predictions for the new season:")
print(final_predictions[final_predictions['GW'] == final_predictions['GW'].min()].sort_values('expected_minutes', ascending=False).head(10))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': final_model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))