import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load model
model = joblib.load('random_forest_model.joblib')

# Streamlit page configuration
st.set_page_config(
    page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #2e86de;
        color: white;
    }
    .stButton>button {
        background-color: #2e86de;
        color: white;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1c5eb6;
        color: white;
    }
    h1 {
        color: #1c5eb6;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and input instructions
st.title("🏠 House Price Prediction")
st.markdown("Enter the features of the house to predict its price:")

# Sidebar header for input features
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
furnishing = st.sidebar.selectbox('Furnishing', ['Fully Furnished', 'Partly Furnished', 'Unfurnished'])
property_category = st.sidebar.selectbox('Property Category', ['High Rise Luxury', 'High Rise Usual', 'Landed Luxury', 'Landed Usual'])
g_size = st.sidebar.selectbox('g_size', ['b.400 - 600', 'c.600 - 1000', 'd.> 1000'])
distance_range = st.sidebar.selectbox('Distance Range', ['< 500m', '< 1km', '< 2km', '< 3km', '< 4km', '< 5km', 'no train station nearby'])
size_category = st.sidebar.selectbox('Size Category', ['Tiny (400-1000 sq ft)', 'Small (1000-1500 sq ft)', 'Medium (1500-2000 sq ft)', 'Large (2000-3000 sq ft)', 'Very Large (3000-5000 sq ft)', 'Huge (>5000 sq ft)'])

# Predict button
if st.button("Predict"):
    # Prepare the user input data
    user_input = [
        location, property_type, furnishing, property_category, g_size,
        distance_range, size_category
    ]
    numeric_data = {
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
    }

    # Encode categorical data
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform([user_input])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    # Combine numeric and encoded data
    numeric_df = pd.DataFrame([numeric_data])
    final_input = pd.concat([numeric_df, encoded_df], axis=1)

    # Standardize numeric features
    scaler = StandardScaler()
    numeric_columns = list(numeric_data.keys())
    final_input[numeric_columns] = scaler.fit_transform(final_input[numeric_columns])

    # Predict house price
    prediction = model.predict(final_input)
    st.success(f"Estimated House Price: RM {prediction[0]:,.2f}")
