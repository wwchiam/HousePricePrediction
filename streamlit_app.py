import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define Min-Max Normalization function
def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Load model and feature names
def load_model_and_features():
    model = joblib.load('random_forest_model.joblib')
    features = joblib.load('feature_names.pkl')  # Ensure this contains the correct feature names
    return model, feature

# Preprocess user input data
def preprocess_input(input_data, feature_names):
    normalization_ranges = {
        'Rooms': (0, 20),
        'Bathrooms': (0, 20),
        'Car Parks': (0, 10),
        'Size': (100, 10000),
        'Distance to Hospital (KM)': (0, 50),
        'Distance to Shopping_mall (KM)': (0, 50),
        'Distance to Train_station (KM)': (0, 50),
        'Distance to Primary_school (KM)': (0, 50),
        'Distance to Secondary_school (KM)': (0, 50),
        'Distance to University (KM)': (0, 50),
    }

    # Initialize a DataFrame with zeros for all feature names
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Update the DataFrame with user inputs and normalize numeric values
    for col, value in input_data.items():
        if col in normalization_ranges:
            min_val, max_val = normalization_ranges[col]
            input_df[col] = min_max_normalize(value, min_val, max_val)
        elif col in input_df.columns:
            input_df[col] = value

    return input_df

def main():
    # Header Section
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏡 House Price Prediction App</h1>", unsafe_allow_html=True)
    #st.image("https://via.placeholder.com/1200x400.png?text=House+Price+Prediction", use_column_width=True)

    model, feature_names = load_model_and_features()

    st.header("User Inputs")

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    user_inputs = {}

    # Numeric Inputs in Column 1
    with col1:
        st.markdown("### 🧮 Numeric Inputs")
        user_inputs['Rooms'] = st.number_input("Rooms", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        user_inputs['Bathrooms'] = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
        user_inputs['Car Parks'] = st.number_input("Car Parks", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        user_inputs['Size'] = st.number_input("Size (sq ft)", min_value=100.0, max_value=10000.0, value=1500.0, step=1.0)
        user_inputs['Distance to Hospital (KM)'] = st.number_input("Distance to Hospital (KM)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        user_inputs['Distance to Shopping_mall (KM)'] = st.number_input("Distance to Shopping Mall (KM)", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
        user_inputs['Distance to Train_station (KM)'] = st.number_input("Distance to Train Station (KM)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        user_inputs['Distance to Primary_school (KM)'] = st.number_input("Distance to Primary School (KM)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        user_inputs['Distance to Secondary_school (KM)'] = st.number_input("Distance to Secondary School (KM)", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
        user_inputs['Distance to University (KM)'] = st.number_input("Distance to University (KM)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)

    # Categorical Inputs in Column 2
    with col2:
        st.markdown("### 📋 Categorical Inputs")
        location = st.selectbox("Location", [
            'alam damai', 'ampang', 'bandar damai perdana', 'bandar menjalara', 'bangsar', 'bangsar south',
            'batu caves', 'brickfields', 'bukit bintang', 'bukit damansara', 'bukit jalil', 'bukit kiara',
            'bukit ledang', 'bukit tunku', 'chan sow lin', 'cheras', 'country heights damansara',
            'damansara', 'desa pandan', 'desa parkcity', 'desa petaling', 'federal hill', 'gombak',
            'happy garden', 'jalan ipoh', 'jalan kuching', 'jalan sultan ismail', 'kepong', 'keramat',
            'kl eco city', 'kl sentral', 'klcc', 'kota damansara', 'kuala lumpur', 'kuchai lama',
            'mid valley city', 'mont kiara', 'off gasing indah,', 'oug', 'pandan indah', 'pandan jaya',
            'pandan perdana', 'pantai', 'puchong', 'rawang', 'salak selatan', 'segambut', 'semarak',
            'sentul', 'seputeh', 'setapak', 'setiawangsa', 'sri damansara', 'sri hartamas', 'sri petaling',
            'sungai besi', 'sungai penchala', 'sunway spk', 'taman connaught', 'taman desa', 'taman duta',
            'taman melati', 'taman melawati', 'taman tun dr ismail', 'taman wangsa permai', 'titiwangsa',
            'wangsa maju'])
        property_type = st.selectbox("Property Type", [
            '1-sty Terrace/Link House', '1.5-sty Terrace/Link House', '2-sty Terrace/Link House',
            '2.5-sty Terrace/Link House', '3-sty Terrace/Link House', '3.5-sty Terrace/Link House',
            '4-sty Terrace/Link House', '4.5-sty Terrace/Link House', 'Apartment', 'Bungalow',
            'Bungalow Land', 'Cluster House', 'Condominium', 'Flat', 'Residential Land',
            'Semi-detached House', 'Serviced Residence', 'Townhouse'])
        furnishing = st.selectbox("Furnishing", ['Fully Furnished', 'Partly Furnished', 'Unfurnished'])
        size_type = st.selectbox("Size Type", ['Built-up', 'Land area'])
        user_inputs['Size_type'] = 0 if size_type == 'Built-up' else 1
        property_category = st.selectbox("Property Category", [
            'High Rise Luxury', 'High Rise Usual', 'Landed Luxury', 'Landed Usual'])
        g_size = st.selectbox("Group Size", ['b.400 - 600', 'c.600 - 1000', 'd.> 1000'])
        distance_range = st.selectbox("Distance Range", ['< 500m', '< 1km', '< 2km', '< 3km', '< 4km', '< 5km', 'no train station nearby'])
        size_category = st.selectbox("Size Category", ['Tiny (400-1000 sq ft)', 'Small (1000-1500 sq ft)', 'Medium (1500-2000 sq ft)', 'Large (2000-3000 sq ft)', 'Very Large (3000-5000 sq ft)', 'Huge (>5000 sq ft)'])

        user_inputs[f"Location_{location}"] = 1
        user_inputs[f"Property Type_{property_type}"] = 1
        user_inputs[f"Furnishing_{furnishing}"] = 1
        user_inputs[f"Property Category_{property_category}"] = 1
        user_inputs[f"g_size_{g_size}"] = 1
        user_inputs[f"Distance Range_{distance_range}"] = 1
        user_inputs[f"Size_Category_{size_category}"] = 1

    st.markdown("### 🌟 User Inputs")
    st.json(user_inputs)

    preprocessed_input = preprocess_input(user_inputs, feature_names)
    st.markdown("### 🔄 Preprocessed Input Data")
    st.write(preprocessed_input)

    if st.button("💡 Predict Price"):
        prediction = model.predict(preprocessed_input)[0]
        st.success(f"The predicted house price is: RM {prediction:,.2f}")

if __name__ == "__main__":
    main()
