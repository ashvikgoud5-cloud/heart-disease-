import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('heart_dataset/dataset.csv')  # Make sure your dataset is in the same folder or provide the correct path

# Assume the target variable is 'target' and all other columns are features
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model as a .pkl file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model has been saved as 'model.pkl'")
