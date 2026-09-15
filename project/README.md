
**Project Directory Structure**

```text
project-root/
├── data/
│   └── employee_data.csv
├── models/
│   └── model.pkl (generated after training)
├── main.py
├── requirements.txt
└── README.md

```

---

**File: `requirements.txt**`

```text
pandas==2.2.0
scikit-learn==1.4.0
numpy==1.26.3

```

---

**File: `data/employee_data.csv**`

```csv
age,years_at_company,monthly_income,satisfaction_score,left
35,5,5000,0.8,0
28,1,3000,0.4,1
42,10,9000,0.9,0
31,3,4500,0.5,1
45,12,11000,0.7,0
25,2,2800,0.3,1
38,7,7500,0.85,0
29,1,3200,0.45,1

```

---

**File: `main.py**`

```python
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

```

---

**File: `README.md**`

```markdown
# Employee Attrition Prediction CLI Tool

A command-line interface machine learning tool built with Python, Pandas, and Scikit-Learn to predict employee attrition risk based on demographic and professional metrics.

## Prerequisites
* Python 3.10 or higher
* Pip package manager

## Environment Setup and Installation
1. Clone the repository and navigate to the root directory:
   ```bash
   git clone [https://github.com/](https://github.com/){github-username}/{repo-name}.git
   cd {repo-name}

```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate

```


3. Install dependencies:
```bash
pip install -r requirements.txt

```



## Configuration & Data

The project uses CSV data located at `data/employee_data.csv`. The dataset includes:

* `age`: Employee age (numeric)
* `years_at_company`: Tenure in years (numeric)
* `monthly_income`: Monthly salary (numeric)
* `satisfaction_score`: Employee rating between 0 and 1 (float)
* `left`: Target classification label (0 or 1)

## Execution Instructions

### 1. Train the Model

Execute the training subcommand via the terminal to fit the classification algorithm and serialize the output model artifact:

```bash
python main.py train --data data/employee_data.csv --model models/model.pkl

```

### 2. Run Predictions

Execute inference on input data records using the generated model:

```bash
python main.py predict --model models/model.pkl --input data/employee_data.csv

```

```

```