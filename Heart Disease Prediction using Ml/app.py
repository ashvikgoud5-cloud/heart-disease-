from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load the dataset
df = pd.read_csv('heart.csv')  # Ensure dataset is in the same folder or provide correct path

# Assume the target variable is 'target' and all other columns are features
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

# Save the trained Random Forest model
with open('rf_model.pkl', 'wb') as rf_model_file:
    pickle.dump(rf_model, rf_model_file)

# Bat Algorithm (Placeholder)
def train_bat_algorithm():
    # Simulate training (replace with actual implementation)
    return np.random.rand()

bat_algorithm_accuracy = train_bat_algorithm()
print(f'Bat Algorithm Accuracy: {bat_algorithm_accuracy * 100:.2f}%')

# Bee Algorithm (Placeholder)
def train_bee_algorithm():
    # Simulate training (replace with actual implementation)
    return np.random.rand()

bee_algorithm_accuracy = train_bee_algorithm()
print(f'Bee Algorithm Accuracy: {bee_algorithm_accuracy * 100:.2f}%')

# Flask app setup
app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'

# Load the trained models
def load_rf_model():
    with open('rf_model.pkl', 'rb') as rf_model_file:
        rf_model = pickle.load(rf_model_file)
    return rf_model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    return render_template('signup_success.html', username=username)

@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form.get('username')
    password = request.form.get('password')
    session['username'] = username
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

        # Load Random Forest model and make prediction
        model = load_rf_model()
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction[0])

    return render_template('predict.html')

@app.route('/run_rf')
def run_rf():
    return render_template('rf_algorithm.html', accuracy=rf_accuracy * 100)

@app.route('/run_bat')
def run_bat():
    return render_template('bat_algorithm.html', accuracy=bat_algorithm_accuracy * 100)

@app.route('/run_bee')
def run_bee():
    return render_template('bee_algorithm.html', accuracy=bee_algorithm_accuracy * 100)


if __name__ == '__main__':
    app.run(debug=True)
