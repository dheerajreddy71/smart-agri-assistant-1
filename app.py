import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Streamlit app
st.set_page_config(page_title="Agriculture Prediction", layout="wide", page_icon="🌾")

# Add a background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://github.com/dheerajreddy71/Webbuild/raw/main/background.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Smart Agri Assistant")

# Load and prepare datasets for yield prediction
yield_df = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/yield_df.csv")
crop_recommendation_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/Crop_recommendation.csv")

yield_preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)
yield_X = yield_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
yield_y = yield_df['hg/ha_yield']
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X, yield_y, train_size=0.8, random_state=0, shuffle=True)
yield_X_train_dummy = yield_preprocessor.fit_transform(yield_X_train)
yield_X_test_dummy = yield_preprocessor.transform(yield_X_test)
yield_model = KNeighborsRegressor(n_neighbors=5)
yield_model.fit(yield_X_train_dummy, yield_y_train)

crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

# Load crop data and train the model for temperature prediction
data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds1.csv", encoding='ISO-8859-1')
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

# Function to predict temperature and humidity requirements for a crop
def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = data[data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None  # Handle cases where crop_name is not found
    predicted_temperature = model.predict(crop_data)
    crop_row = data[data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

# Function to get pest warnings for a crop
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

# Read data from the CSV file and store it in dictionaries
pest_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds2.csv")
for _, row in pest_data.iterrows():
    crop = row[0].strip().lower()
    pest = row[1].strip()
    crop_pest_data[crop] = pest
    planting_time_info[crop] = row[5].strip()
    growth_stage_info[crop] = row[6].strip()
    pesticides_info[crop] = row[4].strip()

def predict_pest_warnings(crop_name):
    crop_name = crop_name.lower()
    specified_crops = [crop_name]

    pest_warnings = []

    for crop in specified_crops:
        if crop in crop_pest_data:
            pests = crop_pest_data[crop].split(', ')
            warning_message = f"\nBeware of pests like {', '.join(pests)} for {crop.capitalize()}.\n"

            if crop in planting_time_info:
                planting_time = planting_time_info[crop]
                warning_message += f"\nPlanting Time: {planting_time}\n"

            if crop in growth_stage_info:
                growth_stage = growth_stage_info[crop]
                warning_message += f"\nGrowth Stages of Plant: {growth_stage}\n"

            if crop in pesticides_info:
                pesticides = pesticides_info[crop]
                warning_message += f"\nUse Pesticides like: {pesticides}\n"
                
            pest_warnings.append(warning_message)

    return '\n'.join(pest_warnings)

# Load and preprocess crop price data
price_data = pd.read_csv('https://github.com/dheerajreddy71/Design_Project/raw/main/pred_data.csv', encoding='ISO-8859-1')
price_data['arrival_date'] = pd.to_datetime(price_data['arrival_date'])
price_data['day'] = price_data['arrival_date'].dt.day
price_data['month'] = price_data['arrival_date'].dt.month
price_data['year'] = price_data['arrival_date'].dt.year
price_data.drop(['arrival_date'], axis=1, inplace=True)

price_X = price_data.drop(['min_price', 'max_price', 'modal_price'], axis=1)
price_y = price_data[['min_price', 'max_price', 'modal_price']]

price_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['state', 'district', 'market', 'commodity', 'variety'])
    ],
    remainder='passthrough'
)

price_X_encoded = price_encoder.fit_transform(price_X)
price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(price_X_encoded, price_y, test_size=0.2, random_state=42)

price_model = LinearRegression()
price_model.fit(price_X_train, price_y_train)

# Fertilizer Recommendation
fertilizer_data = pd.read_csv('https://github.com/dheerajreddy71/Design_Project/raw/main/fertilizer_recommendation.csv', encoding='ISO-8859-1')
fertilizer_data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)
fertilizer_data.dropna(inplace=True)

# Encode categorical variables
encode_soil = LabelEncoder()
fertilizer_data.Soil_Type = encode_soil.fit_transform(fertilizer_data.Soil_Type)

encode_crop = LabelEncoder()
fertilizer_data.Crop_Type = encode_crop.fit_transform(fertilizer_data.Crop_Type)

encode_ferti = LabelEncoder()
fertilizer_data.Fertilizer = encode_ferti.fit_transform(fertilizer_data.Fertilizer)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(fertilizer_data.drop('Fertilizer', axis=1), fertilizer_data.Fertilizer, test_size=0.2, random_state=1)

# Train a Random Forest Classifier
rand = RandomForestClassifier()
rand.fit(x_train, y_train)

# Yield Prediction
st.header("Predict Crop Yield")
year = st.number_input("Year for Yield Prediction", min_value=2000, max_value=2100, value=2024)
rainfall = st.number_input("Average Rainfall (mm per year) for Yield Prediction")
pesticides = st.number_input("Pesticides Used (tonnes) for Yield Prediction")
temp = st.number_input("Average Temperature (°C) for Yield Prediction")
area = st.text_input("Area for Yield Prediction")
item = st.text_input("Item for Yield Prediction")

if st.button("Predict Yield"):
    features = {
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': temp,
        'Area': area,
        'Item': item,
    }
    features_array = np.array([[features['Year'], features['average_rain_fall_mm_per_year'],
                                features['pesticides_tonnes'], features['avg_temp'],
                                features['Area'], features['Item']]], dtype=object)
    transformed_features = yield_preprocessor.transform(features_array)
    predicted_yield = yield_model.predict(transformed_features).reshape(1, -1)
    st.success(f"The predicted yield is {predicted_yield[0][0]:.2f} hectograms (hg) per hectare (ha).")

# Crop Recommendation
st.header("Crop Recommendation")
N = st.number_input("Nitrogen content (N) in soil", min_value=0, max_value=100, value=0)
P = st.number_input("Phosphorus content (P) in soil", min_value=0, max_value=100, value=0)
K = st.number_input("Potassium content (K) in soil", min_value=0, max_value=100, value=0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)

if st.button("Recommend Crop"):
    crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_prediction = crop_model.predict(crop_features)
    st.success(f"Recommended crop is {crop_prediction[0]}.")

    # Get requirements and pest warnings for the recommended crop
    humidity_req, temperature_req = predict_requirements(crop_prediction[0])
    pest_warnings = predict_pest_warnings(crop_prediction[0])

    if humidity_req and temperature_req:
        st.info(f"Temperature Required for {crop_prediction[0]}: {temperature_req:.2f} °F")
        st.info(f"Humidity Required for {crop_prediction[0]}: {humidity_req:.2f} %")
    else:
        st.warning(f"Requirements for {crop_prediction[0]} not found in the dataset.")
    
    if pest_warnings:
        st.info(f"Pest Warnings for {crop_prediction[0]}:\n{pest_warnings}")
    else:
        st.warning(f"Pest warnings for {crop_prediction[0]} not found in the dataset.")

# Price Prediction
st.header("Predict Crop Price")
state = st.selectbox("Select State", price_data['state'].unique())
district = st.selectbox("Select District", price_data['district'].unique())
market = st.selectbox("Select Market", price_data['market'].unique())
commodity = st.selectbox("Select Commodity", price_data['commodity'].unique())
variety = st.selectbox("Select Variety", price_data['variety'].unique())
day = st.number_input("Day of Arrival", min_value=1, max_value=31, value=1)
month = st.number_input("Month of Arrival", min_value=1, max_value=12, value=1)
year = st.number_input("Year of Arrival", min_value=2000, max_value=2100, value=2024)

if st.button("Predict Price"):
    price_features = pd.DataFrame([[state, district, market, commodity, variety, day, month, year]], 
                                  columns=['state', 'district', 'market', 'commodity', 'variety', 'day', 'month', 'year'])
    price_features_encoded = price_encoder.transform(price_features)
    predicted_price = price_model.predict(price_features_encoded)
    st.success(f"Predicted Min Price: {predicted_price[0][0]:.2f}")
    st.success(f"Predicted Max Price: {predicted_price[0][1]:.2f}")
    st.success(f"Predicted Modal Price: {predicted_price[0][2]:.2f}")

# Fertilizer Recommendation
st.header("Fertilizer Recommendation")
temperature = st.number_input("Temperature (°C) for Fertilizer Recommendation", min_value=0.0, max_value=50.0, value=0.0)
humidity = st.number_input("Humidity (%) for Fertilizer Recommendation", min_value=0.0, max_value=100.0, value=0.0)
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=0.0)
soil_type = st.selectbox("Select Soil Type", encode_soil.classes_)
crop_type = st.selectbox("Select Crop Type", encode_crop.classes_)
nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=0)
potassium = st.number_input("Potassium (K)", min_value=0, max_value=100, value=0)
phosphorous = st.number_input("Phosphorous (P)", min_value=0, max_value=100, value=0)

if st.button("Recommend Fertilizer"):
    fertilizer_features = np.array([[temperature, humidity, soil_moisture, 
                                     encode_soil.transform([soil_type])[0], 
                                     encode_crop.transform([crop_type])[0], 
                                     nitrogen, potassium, phosphorous]])
    fertilizer_prediction = rand.predict(fertilizer_features)
    fertilizer_name = encode_ferti.inverse_transform(fertilizer_prediction)[0]
    st.success(f"Recommended Fertilizer: {fertilizer_name}")
