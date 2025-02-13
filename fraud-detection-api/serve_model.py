from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (Make sure the model is saved as 'Fraud Data_Random Forest.pkl')
model = joblib.load('Fraud Data_Random Forest.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize scalers
scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Engineered features function
def create_engineered_features(data):
    # Convert 'signup_time' and 'purchase_time' to datetime
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    # Calculate 'signup_to_purchase_seconds'
    data['signup_to_purchase_seconds'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds()
    
    # Extract 'hour_of_day' and 'day_of_week'
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    
    # 'transaction_count' (example logic, you could aggregate or count based on 'user_id')
    data['transaction_count'] = 1  # In this case, it's a single transaction per user
    
    return data

# Preprocess the data (scaling, encoding)
def preprocess_data(df):
    # Ensure all necessary columns are processed, including categorical ones
    df["Amount"] = scaler.fit_transform(df[["Amount"]])  # Scale 'Amount'

    # Encode categorical columns
    categorical_columns = ["source", "browser", "sex", "country"]
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    
    # Apply StandardScaler to selected features
    columns_to_scale = ['purchase_value', 'hour_of_day', 'day_of_week', 'signup_to_purchase_seconds', 'transaction_count', 'age']
    df[columns_to_scale] = standard_scaler.fit_transform(df[columns_to_scale])
    
    return df

# Load the CSV files into DataFrames
df = pd.read_csv('Fraud_Data.csv')  
df1 = pd.read_csv('IpAddress_to_Country.csv')

# Extract unique values for dropdowns
countries = df1['country'].unique()
sources = df['source'].unique()
browsers = df['browser'].unique()

@app.route('/')
def home():
    # Pass the dropdown data to the template
    return render_template('index.html', countries=countries, sources=sources, browsers=browsers)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    user_id = request.form['user_id']
    signup_time = request.form['signup_time']
    purchase_time = request.form['purchase_time']
    purchase_value = request.form['purchase_value']
    device_id = request.form['device_id']
    source = request.form['source']
    browser = request.form['browser']
    sex = request.form['sex']
    age = request.form['age']
    ip_address = request.form['ip_address']
    Amount = request.form['Amount']
    country = request.form['country']

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'user_id': [user_id],
        'signup_time': [signup_time],
        'purchase_time': [purchase_time],
        'purchase_value': [purchase_value],
        'device_id': [device_id],
        'source': [source],
        'browser': [browser],
        'sex': [sex],
        'age': [age],
        'ip_address': [ip_address],
        'Amount': [Amount],
        'country': [country],
    })

    # Generate engineered features
    input_data = create_engineered_features(input_data)
    
    # Preprocess the input data (scale and encode)
    input_data = preprocess_data(input_data)
    input_data = input_data.drop(columns=['device_id'])  # Drop device_id as it's not used for prediction

    # Select the same features used for training the model
    def ip_to_int(ip):
        parts = ip.split('.')
        return int(parts[0]) * (256 ** 3) + int(parts[1]) * (256 ** 2) + int(parts[2]) * 256 + int(parts[3])

    # Apply this function to the ip_address column
    input_data['ip_address'] = input_data['ip_address'].apply(ip_to_int)

    features = input_data[['user_id', 'purchase_value', 'source', 'browser', 'sex', 'age',
       'ip_address', 'country', 'transaction_count', 'hour_of_day',
       'day_of_week', 'signup_to_purchase_seconds']]

   

    # Model prediction (ensure the model expects the same features)
    prediction = model.predict(features)

    # Return prediction result
    if prediction == 1:
        return render_template('index.html', prediction_text='Fraud Detected', countries=countries, sources=sources, browsers=browsers)
    else:
        return render_template('index.html', prediction_text='No Fraud Detected', countries=countries, sources=sources, browsers=browsers)

if __name__ == '__main__':
    app.run(debug=True)
