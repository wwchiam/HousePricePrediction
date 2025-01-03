import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load model and feature names
model = joblib.load('random_forest_model.joblib')
features = joblib.load('feature_names.pkl')  # Ensure this contains the correct feature names

# Streamlit page configuration
st.set_page_config(
    page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded"
)

# Title and input instructions
st.title("🏠 House Price Prediction")
st.markdown("Enter the features of the house to predict its price:")

# Sidebar header for input features
input_data = {}
st.sidebar.header("Input Features")

# Define the options for categorical features
location_options = [
    "alam damai", "ampang", "bandar damai perdana", "bandar menjalara",
    "bangsar", "bangsar south", "batu caves", "brickfields",
    "bukit bintang", "bukit damansara", "bukit jalil", "bukit kiara",
    "bukit ledang", "bukit tunku", "chan sow lin", "cheras",
    "country heights damansara", "damansara", "desa pandan",
    "desa parkcity", "desa petaling", "federal hill", "jalan ipoh",
    "jalan kuching", "jalan sultan ismail", "kepong", "keramat",
    "kl eco city", "kl sentral", "klcc", "kota damansara",
    "kuala lumpur", "kuchai lama", "mid valley city", "mont kiara",
    "off gasing indah", "oug", "pandan indah", "pandan jaya",
    "pandan perdana", "pantai", "puchong", "rawang", "salak selatan",
    "segambut", "semarak", "sentul", "seputeh", "setapak",
    "setiawangsa", "sri damansara", "sri hartamas", "sri petaling",
    "sungai besi", "sungai penchala", "sunway spk", "taman desa",
    "taman duta", "taman melati", "titiwangsa", "wangsa maju"
]

property_type_options = [
    "1-sty Terrace/Link House", "1.5-sty Terrace/Link House",
    "2-sty Terrace/Link House", "2.5-sty Terrace/Link House",
    "3-sty Terrace/Link House", "3.5-sty Terrace/Link House",
    "4-sty Terrace/Link House", "4.5-sty Terrace/Link House",
    "Apartment", "Bungalow", "Bungalow Land", "Cluster House",
    "Condominium", "Residential Land", "Semi-detached House",
    "Serviced Residence", "Townhouse"
]

# Sidebar inputs for the top 10 features
location = st.sidebar.selectbox("Location", location_options)
property_type = st.sidebar.selectbox("Property Type", property_type_options)
rooms = st.sidebar.slider("Rooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
car_parks = st.sidebar.slider("Car Parks", 0, 5, 1)
size = st.sidebar.number_input("Size (sq ft)", value=1000)
distance_train = st.sidebar.number_input("Distance to Train Station (KM)", value=1.0)
distance_university = st.sidebar.number_input("Distance to University (KM)", value=2.5)
distance_secondary_school = st.sidebar.number_input("Distance to Secondary School (KM)", value=1.8)
distance_hospital = st.sidebar.number_input("Distance to Hospital (KM)", value=1.5)
distance_mall = st.sidebar.number_input("Distance to Shopping Mall (KM)", value=2.0)
distance_primary_school = st.sidebar.number_input("Distance to Primary School (KM)", value=1.2)

# Define label encoder for Size_type
le = LabelEncoder()

# Predict button
if st.button("Predict"):
    # Prepare the user input data for the top 10 features
    user_input = {
        "Rooms": rooms,
        "Bathrooms": bathrooms,
        "Car Parks": car_parks,
        "Size": size,
        "Distance to Train_station (KM)": distance_train,
        "Distance to University (KM)": distance_university,
        "Distance to Secondary_school (KM)": distance_secondary_school,
        "Distance to Hospital (KM)": distance_hospital,
        "Distance to Shopping_mall (KM)": distance_mall,
        "Distance to Primary_school (KM)": distance_primary_school,
        "Size_type": le.fit_transform([property_type])[0],  # Example of label encoding on a feature
    }

    # Initialize a dictionary for the categorical features to be one-hot encoded
    categorical_features = {}

    # One-hot encode the location and property type as done in the model
    categorical_features[f"Location_{location}"] = 1
    categorical_features[f"Property Type_{property_type}"] = 1
    categorical_features[f"Furnishing_{st.sidebar.selectbox('Furnishing', ['Fully Furnished', 'Partly Furnished', 'Unfurnished'])}"] = 1
    categorical_features[f"Property Category_{st.sidebar.selectbox('Property Category', ['High Rise Luxury', 'High Rise Usual', 'Landed Luxury', 'Landed Usual'])}"] = 1
    categorical_features[f"g_size_{st.sidebar.selectbox('g_size', ['b.400 - 600', 'c.600 - 1000', 'd.> 1000'])}"] = 1
    categorical_features[f"Distance Range_{st.sidebar.selectbox('Distance Range', ['< 500m', '< 1km', '< 2km', '< 3km', '< 4km', '< 5km', 'no train station nearby'])}"] = 1
    categorical_features[f"Size_Category_{st.sidebar.selectbox('Size Category', ['Tiny (400-1000 sq ft)', 'Small (1000-1500 sq ft)', 'Medium (1500-2000 sq ft)', 'Large (2000-3000 sq ft)', 'Very Large (3000-5000 sq ft)', 'Huge (>5000 sq ft)'])}"] = 1

    # Add categorical features to the user input data
    user_input.update(categorical_features)

    # Create a DataFrame from the user input
    user_input_encoded = pd.DataFrame([user_input])

    # Initialize the aligned input with zeros for all columns
    aligned_input = pd.DataFrame(0, index=[0], columns=features)

    # Ensure all input data matches the expected columns
    for col in user_input_encoded.columns:
        if col in aligned_input.columns:
            aligned_input[col] = user_input_encoded[col]

    # Reorder the aligned input to match the feature order expected by the model
    aligned_input = aligned_input[features]

    # List of numeric features for scaling (ensure these match with the model's columns)
    numeric_features = [
        "Rooms", "Bathrooms", "Car Parks", "Size", 
        "Distance to Hospital (KM)", "Distance to Shopping_mall (KM)", 
        "Distance to Train_station (KM)", "Distance to Primary_school (KM)", 
        "Distance to Secondary_school (KM)", "Distance to University (KM)"
    ]

    # Apply StandardScaler to the numeric features
    scaler = StandardScaler()
    aligned_input[numeric_features] = scaler.fit_transform(aligned_input[numeric_features])

    # Make prediction
    predicted_price = model.predict(aligned_input)[0]

    # Display the results
    st.subheader("Predicted House Price:")
    st.write(f"<span style='font-size:24px; color:#1c5eb6; font-weight:bold;'>RM {predicted_price:,.2f}</span>", unsafe_allow_html=True)
    st.write("### Input Data for Prediction:")
    st.dataframe(aligned_input.style.set_properties(**{'background-color': '#f7f9fc', 'color': '#333', 'border': '1px solid #ccc'}))
