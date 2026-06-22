import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

# 1. Load the data from your folder
df = pd.read_csv('data/churn.csv')

# 2. Clean the data (Machine learning only understands numbers, not text!)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(['customerID', 'Churn'], axis=1)
X = pd.get_dummies(X, drop_first=True) # Turns words like "Male"/"Female" into 0s and 1s
y = df['Churn']

# 3. Split the data into a "study guide" and a "final test"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Start MLflow (our tracker) and train the model!
with mlflow.start_run():
    # Create the model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3)
    # Train the model on the study guide
    model.fit(X_train, y_train)

    # Make predictions on the final test
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 5. Log the results to our tracking dashboard
    mlflow.log_metric("accuracy", accuracy)
    mlflow.xgboost.log_model(model, "xgboost-model")

    print(f"Success! Model trained with an accuracy of: {accuracy}")
model.save_model("src/xgboost_model.json")