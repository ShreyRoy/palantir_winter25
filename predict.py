import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the data
file_with_target = 'forest_health_data_with_target.csv'
file_without_target = 'forest_health_data.csv'
data = pd.read_csv(file_with_target)
data_to_predict = pd.read_csv(file_without_target)

# Preprocessing
# Drop unnecessary columns
X = data.drop(columns=['Plot_ID', 'Health_Status'])
y = data['Health_Status']

# Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature Importance
importances = rf_model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, align='center')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Predict on new data
data['Predicted_Health_Status'] = le.inverse_transform(rf_model.predict(X))
data[['Plot_ID', 'Health_Status', 'Predicted_Health_Status']].to_csv('predicted_tree_health_same_dataset.csv', index=False)
print("Predictions saved to 'predicted_tree_health_same_dataset.csv'")

from joblib import dump, load
dump(rf_model, 'model.pkl')
dump(le, 'label_encoder.pkl')
