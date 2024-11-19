# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
print("Loading dataset...")
data = pd.read_csv("C:\\Users\\anura\\Desktop\\Project 3- Iris classification\\Data\\Iris.csv")  
print("Dataset loaded successfully!")

# Step 2: Label Encoding for the target column
print("Encoding target column...")
label_encoder = LabelEncoder()
data["Species_encoded"] = label_encoder.fit_transform(data["Species"])

# Verify encoding
print("Encoded values for Species:\n", data[["Species", "Species_encoded"]].drop_duplicates())

# Step 3: Feature-Target Split
print("Splitting features and target...")
X = data.drop(columns=["Species", "Species_encoded"])  # Input features
y = data["Species_encoded"]  # Encoded target variable

# Step 4: Train-Test Split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize Features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Model
print("Training the Random Forest Classifier...")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training completed!")

# Step 7: Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Save the Model and Scaler
print("Saving the model and scaler...")
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
print("Model and scaler saved successfully!")


print("All tasks completed!")
