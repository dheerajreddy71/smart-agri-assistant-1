import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from googletrans import Translator

# Initialize the translator
translator = Translator()

# Function to translate text
def translate_text(text, lang):
    try:
        translation = translator.translate(text, dest=lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Load and prepare datasets
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

data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds1.csv", encoding='ISO-8859-1')
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = data[data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None
    predicted_temperature = model.predict(crop_data)
    crop_row = data[data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

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

# Streamlit app
st.set_page_config(page_title=translate_text("Smart Agri Assistant", "en"), layout="wide", page_icon="🌾")

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

# Language selection
selected_language = st.selectbox("Select Language", options=['en', 'hi'], format_func=lambda x: {"en": "English", "hi": "Hindi"}[x])

# Translating static text based on selected language
st.title(translate_text("Smart Agri Assistant", selected_language))

# Yield Prediction
st.header(translate_text("Predict Crop Yield", selected_language))
year = st.number_input(translate_text("Year", selected_language), min_value=2000, max_value=2100, value=2024)
rainfall = st.number_input(translate_text("Average Rainfall (mm per year)", selected_language))
pesticides = st.number_input(translate_text("Pesticides Used (tonnes)", selected_language))
temp = st.number_input(translate_text("Average Temperature (°C)", selected_language))
area = st.text_input(translate_text("Area (Country)", selected_language))
item = st.text_input(translate_text("Item (Crop Name)", selected_language))

if st.button(translate_text("Predict Yield", selected_language)):
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
    st.success(translate_text("The predicted yield is {:.2f} hectograms (hg) per hectare (ha).", selected_language).format(predicted_yield[0][0]))

# Crop Recommendation
st.header(translate_text("Recommend Crops", selected_language))
N = st.number_input(translate_text("Nitrogen (N)", selected_language))
P = st.number_input(translate_text("Phosphorus (P)", selected_language))
K = st.number_input(translate_text("Potassium (K)", selected_language))
temperature = st.number_input(translate_text("Temperature (°C)", selected_language))
humidity = st.number_input(translate_text("Humidity (%)", selected_language))
ph = st.number_input(translate_text("Soil pH", selected_language))
rainfall_input = st.number_input(translate_text("Rainfall (mm)", selected_language))

if st.button(translate_text("Recommend Crop", selected_language)):
    crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall_input]])
    recommended_crop = crop_model.predict(crop_features)
    st.success(translate_text("Recommended Crop: {}", selected_language).format(recommended_crop[0]))

# Pest Warnings
st.header(translate_text("Predict Crop Requirements and Pest Warnings", selected_language))
crop_name = st.text_input(translate_text("Crop Name", selected_language))

if st.button(translate_text("Predict Requirements", selected_language)):
    humidity_required, predicted_temperature = predict_requirements(crop_name)
    if humidity_required is not None:
        st.write(translate_text("Required Humidity: {:.2f}%", selected_language).format(humidity_required))
        st.write(translate_text("Predicted Temperature: {:.2f}°C", selected_language).format(predicted_temperature))
    else:
        st.error(translate_text("No data found for the specified crop.", selected_language))
    pest_warnings = predict_pest_warnings(crop_name)
    st.write(pest_warnings)

# Price Prediction
st.header(translate_text("Predict Crop Prices", selected_language))
state = st.text_input(translate_text("State", selected_language))
district = st.text_input(translate_text("District", selected_language))
market = st.text_input(translate_text("Market", selected_language))
commodity = st.text_input(translate_text("Commodity", selected_language))
variety = st.text_input(translate_text("Variety", selected_language))

if st.button(translate_text("Predict Price", selected_language)):
    price_features = np.array([[state, district, market, commodity, variety]])
    price_features_encoded = price_encoder.transform(price_features)
    min_price, max_price, modal_price = price_model.predict(price_features_encoded).flatten()
    st.success(translate_text("Min Price: ₹{:.2f}", selected_language).format(min_price))
    st.success(translate_text("Max Price: ₹{:.2f}", selected_language).format(max_price))
    st.success(translate_text("Modal Price: ₹{:.2f}", selected_language).format(modal_price))
