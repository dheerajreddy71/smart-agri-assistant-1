import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Streamlit app configuration
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
fertilizer_data = pd.read_csv('https://raw.githubusercontent.com/dheerajreddy71/Design_Project/main/fertilizer_recommendation.csv', encoding='utf-8')
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
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
rainfall = st.number_input("Average Rainfall (mm per year)")
pesticides = st.number_input("Pesticides Used (tonnes)")
temp = st.number_input("Average Temperature (°C)")
area = st.text_input("Area")
item = st.text_input("Item")

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
st.header("Recommend Crops")
N = st.number_input("Nitrogen (N) content in soil")
P = st.number_input("Phosphorus (P) content in soil")
K = st.number_input("Potassium (K) content in soil")
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH value of the soil")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Recommend Crop"):
    features = [[N, P, K, temperature, humidity, ph, rainfall]]
    recommended_crop = crop_model.predict(features)[0]
    st.success(f"The recommended crop is {recommended_crop}.")

# Crop Requirements and Pest Warnings
st.header("Crop Requirements and Pest Warnings")
crop_name = st.text_input("Enter crop name")

if st.button("Predict Crop Requirements"):
    if crop_name:
        humidity_required, predicted_temperature = predict_requirements(crop_name)
        if humidity_required and predicted_temperature:
            st.success(f"Humidity required for {crop_name}: {humidity_required} %")
            st.success(f"Predicted temperature required for {crop_name}: {predicted_temperature:.2f} °F")
        else:
            st.error(f"Crop {crop_name} not found in the dataset.")

if st.button("Predict Pest Warnings"):
    if crop_name:
        warnings = predict_pest_warnings(crop_name)
        if warnings:
            st.success(warnings)
        else:
            st.error(f"No pest warnings found for crop {crop_name}.")

# Crop Price Prediction
st.header("Predict Crop Price")
state = st.text_input("State")
district = st.text_input("District")
market = st.text_input("Market")
commodity = st.text_input("Commodity")
variety = st.text_input("Variety")
min_temp = st.number_input("Min Temperature (°C)")
max_temp = st.number_input("Max Temperature (°C)")
rainfall = st.number_input("Rainfall (mm)")
day = st.number_input("Day")
month = st.number_input("Month")
year = st.number_input("Year")

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'state': [state],
        'district': [district],
        'market': [market],
        'commodity': [commodity],
        'variety': [variety],
        'min_temp': [min_temp],
        'max_temp': [max_temp],
        'rainfall': [rainfall],
        'day': [day],
        'month': [month],
        'year': [year]
    })
    input_data_encoded = price_encoder.transform(input_data)
    predicted_prices = price_model.predict(input_data_encoded)
    min_price, max_price, modal_price = predicted_prices[0]
    st.success(f"Predicted Minimum Price: {min_price:.2f}")
    st.success(f"Predicted Maximum Price: {max_price:.2f}")
    st.success(f"Predicted Modal Price: {modal_price:.2f}")

# Fertilizer Recommendation
st.header("Recommend Fertilizer")
soil_type = st.selectbox("Soil Type", encode_soil.classes_)
crop_type = st.selectbox("Crop Type", encode_crop.classes_)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
moisture = st.number_input("Moisture (%)")

if st.button("Recommend Fertilizer"):
    soil_type_encoded = encode_soil.transform([soil_type])[0]
    crop_type_encoded = encode_crop.transform([crop_type])[0]
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Soil_Type': [soil_type_encoded],
        'Crop_Type': [crop_type_encoded]
    })
    predicted_fertilizer = rand.predict(input_data)[0]
    fertilizer_name = encode_ferti.inverse_transform([predicted_fertilizer])[0]
    st.success(f"The recommended fertilizer is {fertilizer_name}.")

