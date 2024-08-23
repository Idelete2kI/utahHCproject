import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances

def find_similar_shots(nhl_csv, my_csv, row_index, feature_weights, top_n):
    # Load datasets
    nhl = nhl_csv
    my = my_csv
    
    # Identify numeric columns 
    numeric_columns_nhl = nhl.select_dtypes(include=['number']).columns
    numeric_columns_my = my.select_dtypes(include=['number']).columns
    
    # Find common numerical columns
    common_numeric_columns = list(set(numeric_columns_nhl) & set(numeric_columns_my))
    
    # Find common categorical columns
    categorical_columns_nhl = [col for col in nhl.columns if col not in numeric_columns_nhl]
    categorical_columns_my = [col for col in my.columns if col not in numeric_columns_my]
    common_categorical_columns = list(set(categorical_columns_nhl) & set(categorical_columns_my))
    
    # Ensure the columns are present in both DataFrames
    common_columns = common_numeric_columns + common_categorical_columns
    
    nhl = nhl[common_columns]
    my = my[common_columns]
    
    # Define features
    numerical_features = common_numeric_columns
    categorical_features = common_categorical_columns
    
    # Create pipelines
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    
    # Fit and transform the NHL dataset and transform the personal shots dataset
    X_nhl_preprocessed = preprocessor.fit_transform(nhl)
    X_my_preprocessed = preprocessor.transform(my)
    
    # Get feature names from one-hot encoder
    feature_names = numerical_features + list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
    
    # Initialize weights
    weights = np.ones(len(feature_names))
    
    # Create a dictionary of weights based on the feature names
    for feature, weight in feature_weights.items():
        if feature in feature_names:
            index = feature_names.index(feature)
            weights[index] = weight
    
    # Apply weights to the features
    X_nhl_weighted = X_nhl_preprocessed * weights
    X_my_weighted = X_my_preprocessed * weights
    
    # Calculate pairwise distances
    distances = pairwise_distances(X_my_weighted, X_nhl_weighted, metric='euclidean')
    
    # Ensure the row index is valid
    if row_index < 0 or row_index >= len(my):
        raise ValueError("Row index is out of range for 'my' DataFrame.")
    
    # Find the indices of the top N most similar shots
    row_distances = distances[row_index]
    top_indices = np.argsort(row_distances)[:top_n]  # Get indices of the top N smallest distances
    
    # Retrieve the top N most similar shots
    g = nhl['goal'].iloc[top_indices]
    
    # Print the top N most similar NHL shots
    
    return float(g.sum())/float(top_n)
    

feature_weights = {
    'timeSinceLastEvent': 3,
    'event': 0,
    'shotGeneratedRebound': 1,
    'xCordAdjusted': 4,
    'yCordAdjusted': 6,
    'shotAngleAdjusted': 13,  
    'shotDistance': 25,       
    'shotType': 7,
    'shotRebound': 6,
    'shotRush': 10,           
    'distanceFromLastEvent': 3,
    'lastEventShotAngle': 3,
    'lastEventShotDistance': 4,
    'lastEventCategory': 2,
    'lastEventxCord_adjusted': 0,
    'lastEventyCord_adjusted': 2,
    'shooterLeftRight': 0, 
    'offWing': 2,
    'isHomeTeam': 0,
    'shotWasOnGoal': 0,
    'shootingTeamDefencemenOnIce': 1,
    'shootingTeamForwardsOnIce': 1,
    'defendingTeamDefencemenOnIce': 2.5,
    'defendingTeamForwardsOnIce': 2,
    'xGoal': 0,
    'goal': 0,
    'shotID': 0,
    'teamCode': 0,
    'game_id': 0,
    'shotOnEmptyNet':2.5
    
}

df = pd.read_csv("C:/Users/neelc/Desktop/shots_2007-2022.csv")

numeric_columns = df.select_dtypes(include=['number']).columns

df_abs = df.copy()

# Apply abs() to numeric columns only
df_abs[numeric_columns] = df_abs[numeric_columns].abs()

df = df_abs


df1 = pd.read_csv("C:/Users/neelc/Desktop/ARIshots_filtered.csv")

numeric_columns = df1.select_dtypes(include=['number']).columns

df1_abs = df1.copy()

# Apply abs() to numeric columns only
df1_abs[numeric_columns] = df1_abs[numeric_columns].abs()

df1 = df1_abs

#Run model through shot data
team = ''
vals= []
sum = 0
for i in range(len(df1)):
    
    chance = find_similar_shots(df, df1, i, feature_weights=feature_weights, top_n=300)
    if df1['shotRebound'].iloc[i] == 1:
        prevShotxG = vals[i-1]
        chance = chance * (1-prevShotxG)
    print(chance)
    vals.append(chance)
    if i + 1 == len(df1):
        game = df1['game_id'].iloc[i]
        print(f"The total non-flurry adjusted expected goals for {team} in game {game} was: {sum}")    
    elif df1['game_id'].iloc[i] == df1['game_id'].iloc[i+1]:
        sum+=chance
    else: 
        game = df1['game_id'].iloc[i]
        print(f"The total non-flurry adjusted expected goals for {team} in game {game} was: {sum}")
        sum = 0
     
    

