"""
Data preprocessing utilities for customer segmentation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    Load and preprocess the customer data.
    """
    # Load data
    print("Loading customer dataset...")
    train_df = pd.read_csv('data/raw/Train.csv')
    
    # Print initial data overview
    print("\nInitial Data Overview:")
    print(train_df.info())
    
    # Print spending score distribution
    print("\nSpending Score Distribution:")
    print(train_df['Spending_Score'].value_counts())
    print("\nUnique Spending Score values:", train_df['Spending_Score'].unique())
    
    # Map spending scores
    spending_map = {'Low': 1, 'Average': 2, 'High': 3}
    train_df['Spending_Score'] = train_df['Spending_Score'].map(spending_map)
    
    print("\nAfter mapping - Spending Score Distribution:")
    print(train_df['Spending_Score'].value_counts())
    
    # Fill missing values with median
    train_df['Work_Experience'].fillna(train_df['Work_Experience'].median(), inplace=True)
    train_df['Family_Size'].fillna(train_df['Family_Size'].median(), inplace=True)
    
    # Select features for clustering
    features = ['Age', 'Work_Experience', 'Family_Size', 'Spending_Score']
    X = train_df[features]
    
    # Print feature statistics
    print("\nFeature Statistics:")
    print(X.describe())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_df, X_scaled
