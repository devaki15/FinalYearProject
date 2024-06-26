import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('sleep_apnea_data.csv')

# Split data into features and target
X = df.drop('SleepApnea', axis=1)
y = df['SleepApnea']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = (accuracy_score(y_test, y_pred))*100
print(f"Model Accuracy: {accuracy}")

# Save the trained model
joblib.dump(model, 'sleep_apnea_model.pkl')
print("Model trained and saved successfully.")
