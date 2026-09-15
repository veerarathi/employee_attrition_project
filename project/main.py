
import os
import argparse
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data_path, model_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    X = df[['age', 'years_at_company', 'monthly_income', 'satisfaction_score']]
    y = df['left']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained successfully. Test Accuracy: {acc:.2f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def predict_data(model_path, input_data_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input data not found at {input_data_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    df = pd.read_csv(input_data_path)
    features = df[['age', 'years_at_company', 'monthly_income', 'satisfaction_score']]
    predictions = model.predict(features)
    
    df['predicted_left'] = predictions
    print("\nPrediction Results:")
    print(df[['age', 'years_at_company', 'monthly_income', 'satisfaction_score', 'predicted_left']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Employee Attrition Prediction CLI Tool")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    train_parser = subparsers.add_parser('train', help='Train the machine learning model')
    train_parser.add_argument('--data', type=str, default='data/employee_data.csv', help='Path to training data CSV')
    train_parser.add_argument('--model', type=str, default='models/model.pkl', help='Path to save trained model')
    
    pred_parser = subparsers.add_parser('predict', help='Run predictions using the trained model')
    pred_parser.add_argument('--model', type=str, default='models/model.pkl', help='Path to trained model')
    pred_parser.add_argument('--input', type=str, default='data/employee_data.csv', help='Path to input CSV for prediction')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data, args.model)
    elif args.command == 'predict':
        predict_data(args.model, args.input)