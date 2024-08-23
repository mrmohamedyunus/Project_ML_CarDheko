import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing steps
model = joblib.load('C:\Guvi\projects\Car_Dekho\Car_Dataset_Structured\car_price_prediction_model.pkl')
label_encoders = joblib.load('C:\Guvi\projects\Car_Dekho\Car_Dataset_Structured\label_encoders.pkl')
scalers = joblib.load('C:\Guvi\projects\Car_Dekho\Car_Dataset_Structured\scalers.pkl')

# Load dataset for filtering and identifying similar data
data = pd.read_csv('C:\Guvi\projects\Car_Dekho\Car_Dataset_Structured\car_dekho_cleaned_dataset_Raw.csv')

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

# Function to filter data based on user selections
def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

# Preprocessing function for user input
def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    df['brand_popularity'] = df['oem'].map(brand_popularity)
    df['mileage_normalized'] = df['mileage'] / df['car_age']

    # Apply label encoding
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0])

    # Apply min-max scaling
    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])

    return df

# Streamlit Application
st.title("Car Price Prediction")

# Sidebar for user inputs
st.sidebar.header('Input Car Features')

# Set background colors
input_background_color = "lightcoral"  # Light maroon color
result_background_color = "#FFF8E7"  # Cosmic latte or beige color

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container {{
        background-color: {result_background_color};
    }}
    .stButton>button {{
        background-color: lightblue;
        color: white;
    }}
    .result-container {{
        text-align: center;
        background-color: {result_background_color};
        padding: 20px;
        border-radius: 10px;
    }}
    .prediction-title {{
        font-size: 28px;
        color: maroon;
    }}
    .prediction-value {{
        font-size: 36px;
        font-weight: bold;
        color: maroon;
    }}
    .info {{
        font-size: 18px;
        color: grey;
    }}
    .sidebar .sidebar-content {{
        background-color: {input_background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Get user inputs with visual representation
def visual_selectbox(label, options, index=0):
    selected_option = st.sidebar.selectbox(label, options, index=index)
    return selected_option

# Get user inputs in a defined order
selected_oem = visual_selectbox('1. Original Equipment Manufacturer (OEM)', data['oem'].unique())

filtered_data = filter_data(oem=selected_oem)
selected_model = visual_selectbox('2. Car Model', filtered_data['model'].unique())

filtered_data = filter_data(oem=selected_oem, model=selected_model)
body_type = visual_selectbox('3. Body Type', filtered_data['bt'].unique())

filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type)
fuel_type = visual_selectbox('4. Fuel Type', filtered_data['ft'].unique())

filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
transmission = visual_selectbox('5. Transmission Type', filtered_data['transmission'].unique())

filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
seat_count = visual_selectbox('6. Seats', filtered_data['Seats'].unique())

filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type, seats=seat_count)
selected_variant = visual_selectbox('7. Variant Name', filtered_data['variantName'].unique())

modelYear = st.sidebar.number_input('8. Year of Manufacture', min_value=1980, max_value=2024, value=2015)

ownerNo = st.sidebar.number_input('9. Number of Previous Owners', min_value=0, max_value=10, value=1)

km = st.sidebar.number_input('10. Kilometers Driven', min_value=0, max_value=500000, value=10000)

# Adjust mileage slider
min_mileage = np.floor(filtered_data['mileage'].min())
max_mileage = np.ceil(filtered_data['mileage'].max())

# Ensure mileage slider has an interval of 0.5
min_mileage = float(min_mileage)
max_mileage = float(max_mileage)

mileage = st.sidebar.slider('11. Mileage (kmpl)', min_value=min_mileage, max_value=max_mileage, value=min_mileage, step=0.5)

city = visual_selectbox('12. City', data['City'].unique())

# Create a DataFrame for user input
user_input_data = {
    'ft': [fuel_type],
    'bt': [body_type],
    'km': [km],
    'transmission': [transmission],
    'ownerNo': [ownerNo],
    'oem': [selected_oem],
    'model': [selected_model],
    'modelYear': [modelYear],
    'variantName': [selected_variant],
    'City': [city],
    'mileage': [mileage],
    'Seats': [seat_count],
    'car_age': [2024 - modelYear],
    'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem)],
    'mileage_normalized': [mileage / (2024 - modelYear)]
}

user_df = pd.DataFrame(user_input_data)

# Ensure the columns are in the correct order and match the trained model's features
user_df = user_df[features]

# Preprocess user input data
user_df = preprocess_input(user_df)

# Button to trigger prediction
if st.sidebar.button('Predict'):
    if user_df.notnull().all().all():
        try:
            # Make prediction
            prediction = model.predict(user_df)
            
            st.markdown(f"""
                <div class="result-container">
                    <h2 class="prediction-title">Predicted Car Price</h2>
                    <p class="prediction-value">₹{prediction[0]:,.2f}</p>
                    <p class="info">Car Age: {user_df['car_age'][0]} years</p>
                    <p class="info">Efficiency Score: {user_df['mileage_normalized'][0]:,.2f} km/year</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
        st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")
