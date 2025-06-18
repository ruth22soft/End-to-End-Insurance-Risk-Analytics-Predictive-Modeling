import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['TransactionMonth'])

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns
    
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Feature engineering
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['VehicleAge'] = df['TransactionMonth'].dt.year - df['RegistrationYear']
    return df

def prepare_model_data(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    cat_cols = X.select_dtypes(include='object').columns
    X_encoded = encoder.fit_transform(X[cat_cols])
    
    # Split data
    return train_test_split(
        pd.concat([X.drop(cat_cols, axis=1), 
                  pd.DataFrame(X_encoded.toarray())], axis=1),
        y,
        test_size=0.2,
        random_state=42
    )