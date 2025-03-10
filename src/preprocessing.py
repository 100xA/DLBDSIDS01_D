import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    train_df = pd.read_csv('../data/raw/Train.csv')
    
    print("\nInitial Data Overview:")
    print(train_df.info())
    
    print("\nSpending Score Distribution:")
    print(train_df['Spending_Score'].value_counts())

    spending_map = {'Low': 1, 'Average': 2, 'High': 3}
    train_df['Spending_Score'] = train_df['Spending_Score'].map(spending_map)
    
    print("\nAfter mapping - Spending Score Distribution:")
    print(train_df['Spending_Score'].value_counts())

    train_df = train_df.fillna({
        'Work_Experience': train_df['Work_Experience'].median(),
        'Family_Size': train_df['Family_Size'].median()
    })
    
    features = ['Age', 'Work_Experience', 'Family_Size', 'Spending_Score']
    X = train_df[features]
    
    print("\nFeature Statistics:")
    print(X.describe())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_df, X_scaled
