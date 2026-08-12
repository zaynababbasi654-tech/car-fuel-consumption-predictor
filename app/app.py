import streamlit as st
import pandas as pd
import joblib
import os
import requests

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Fuel Consumption Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Temporary local location for downloaded model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "fuel_consumption_model.pkl"
)

# Hugging Face model URL
MODEL_URL = (
    "https://huggingface.co/"
    "Zaynababbasi654/car-fuel-consumption-model/"
    "resolve/main/fuel_consumption_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Fuel_consumption_2000-2022.csv"
)

# ============================================================
# DOWNLOAD + LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.info(
            "⏳ Downloading machine learning model... "
            "Please wait."
        )

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=600
        )

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    return joblib.load(MODEL_PATH)


# ============================================================
# CACHED DATASET
# ============================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(DATA_PATH)

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    return data


# ============================================================
# LOAD
# ============================================================

try:

    model = load_model()
    df = load_dataset()

except Exception as e:

    st.error(
        "❌ Could not load model or dataset."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "year",
    "make",
    "model",
    "vehicle class",
    "engine size",
    "transmission",
    "fuel"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ Required columns are missing."
    )

    st.write("Missing:")
    st.write(missing_columns)

    st.write("Available columns:")
    st.write(df.columns.tolist())

    st.stop()


# ============================================================
# PAKISTAN MARKET CARS
# ============================================================

pakistan_cars = {

    "Toyota": [
        "Corolla",
        "Yaris",
        "Camry",
        "Fortuner",
        "Hilux",
        "Land Cruiser",
        "Land Cruiser Prado",
        "Rush",
        "Raize",
        "Vitz"
    ],

    "Honda": [
        "Civic",
        "City",
        "BR-V",
        "HR-V",
        "Vezel"
    ],

    "Suzuki": [
        "Alto",
        "Cultus",
        "Wagon R",
        "Swift",
        "Bolan",
        "Ravi",
        "Mehran",
        "Every",
        "Jimny"
    ],

    "Kia": [
        "Picanto",
        "Stonic",
        "Sportage",
        "Sorento",
        "Carnival"
    ],

    "Hyundai": [
        "Santro",
        "Tucson",
        "Elantra",
        "Sonata",
        "Santa Fe",
        "Porter"
    ],

    "Changan": [
        "Alsvin",
        "Oshan X7",
        "Karvaan",
        "M9"
    ],

    "MG": [
        "HS",
        "ZS",
        "ZS EV",
        "GT",
        "MG 4"
    ],

    "Haval": [
        "H6",
        "Jolion",
        "H6 HEV"
    ],

    "DFSK": [
        "Glory 580",
        "Glory 500",
        "C37"
    ],

    "FAW": [
        "V2",
        "V80",
        "X-PV"
    ],

    "Proton": [
        "Saga",
        "X70"
    ],

    "Prince": [
        "Pearl",
        "K07"
    ],

    "United": [
        "Alpha",
        "Bravo"
    ],

    "BAIC": [
        "BJ40",
        "X25"
    ],

    "Isuzu": [
        "D-Max"
    ]
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚗 Car Fuel Consumption Predictor"
)

st.markdown(
    """
    ### Smart Fuel Efficiency Analysis

    Enter your car details and our Machine Learning model
    will estimate fuel consumption, efficiency and fuel usage.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Input Settings")

    input_mode = st.radio(
        "Choose Input Method",
        [
            "🇵🇰 Pakistani Car List",
            "✍️ Manual Entry"
        ]
    )

    st.divider()

    st.info(
        """
        **Model**

        Random Forest Regression

        Output:
        Fuel Consumption
        + Efficiency Rating
        + Fuel Usage
        + Estimated Cost
        """
    )


# ============================================================
# CAR INFORMATION
# ============================================================

st.subheader("🚘 Car Information")

col1, col2 = st.columns(2)


# ============================================================
# MAKE / MODEL
# ============================================================

if input_mode == "🇵🇰 Pakistani Car List":

    with col1:

        make = st.selectbox(
            "🏭 Car Make",
            sorted(pakistan_cars.keys())
        )

    with col2:

        model_name = st.selectbox(
            "🚗 Car Model",
            pakistan_cars[make]
        )

else:

    with col1:

        make = st.text_input(
            "🏭 Car Make",
            placeholder="Example: Toyota"
        )

    with col2:

        model_name = st.text_input(
            "🚗 Car Model",
            placeholder="Example: Corolla"
        )


# ============================================================
# YEAR / ENGINE
# ============================================================

col3, col4 = st.columns(2)

with col3:

    year = st.number_input(
        "📅 Model Year",
        min_value=1990,
        max_value=2026,
        value=2020,
        step=1
    )

with col4:

    engine_size = st.number_input(
        "🔧 Engine Size (L)",
        min_value=0.5,
        max_value=10.0,
        value=1.8,
        step=0.1
    )


# ============================================================
# VEHICLE CLASS
# ============================================================

vehicle_classes = sorted(
    df["vehicle class"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

vehicle_class = st.selectbox(
    "🚙 Vehicle Class",
    vehicle_classes
)


# ============================================================
# TRANSMISSION
# ============================================================

transmissions = sorted(
    df["transmission"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

transmission = st.selectbox(
    "⚙️ Transmission",
    transmissions
)


# ============================================================
# FUEL TYPE
# ============================================================

st.subheader("⛽ Fuel Type")

fuel_display = st.selectbox(
    "Select Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Gasoline",
        "CNG / Gas",
        "Other / Manual"
    ]
)


# ============================================================
# MAP USER FUEL TO DATASET FUEL
# ============================================================

fuel_mapping = {

    "Petrol": "X",

    "Diesel": "D",

    "Gasoline": "Z",

    "CNG / Gas": "N",

    "Other / Manual": "X"
}


if fuel_display == "Other / Manual":

    fuel = st.text_input(
        "Enter Fuel Code / Type",
        placeholder="Example: X"
    )

else:

    fuel = fuel_mapping[fuel_display]


# ============================================================
# CURRENCY
# ============================================================

st.divider()

st.subheader(
    "💰 Fuel Cost Calculator"
)

currency_option = st.selectbox(
    "Currency",
    [
        "PKR",
        "USD",
        "AED",
        "Other / Manual"
    ]
)


# ============================================================
# CURRENCY SETTINGS
# ============================================================

if currency_option == "Other / Manual":

    custom_currency = st.text_input(
        "Currency Code",
        placeholder="Example: GBP"
    )

    currency_symbol = st.text_input(
        "Currency Symbol",
        placeholder="Example: £"
    )

    fuel_price = st.number_input(
        "Fuel Price per Liter",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

else:

    currency_symbol = {
        "PKR": "PKR",
        "USD": "$",
        "AED": "AED"
    }[currency_option]

    default_prices = {
        "PKR": 270.0,
        "USD": 0.75,
        "AED": 2.75
    }

    fuel_price = st.number_input(
        f"Fuel Price per Liter ({currency_option})",
        min_value=0.0,
        value=default_prices[currency_option],
        step=0.1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Fuel Consumption",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not make.strip() or not model_name.strip():

        st.warning(
            "⚠️ Please enter/select both Make and Model."
        )

        st.stop()


    if (
        fuel_display == "Other / Manual"
        and not fuel.strip()
    ):

        st.warning(
            "⚠️ Please enter your fuel type/code."
        )

        st.stop()


    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    car_data = pd.DataFrame(
        [{
            "year": year,
            "make": make.strip(),
            "model": model_name.strip(),
            "vehicle class": vehicle_class,
            "engine size": engine_size,
            "transmission": transmission,
            "fuel": fuel
        }]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = float(
            model.predict(car_data)[0]
        )

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(e))

        st.stop()


    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    prediction = max(
        prediction,
        0
    )


    # ========================================================
    # EFFICIENCY RATING
    # ========================================================

    if prediction <= 6:

        rating = "Excellent"
        emoji = "🟢"

    elif prediction <= 8:

        rating = "Good"
        emoji = "🟢"

    elif prediction <= 11:

        rating = "Average"
        emoji = "🟡"

    else:

        rating = "Poor"
        emoji = "🔴"


    # ========================================================
    # FUEL USAGE
    # ========================================================

    fuel_100 = prediction
    fuel_500 = prediction * 5
    fuel_1000 = prediction * 10


    # ========================================================
    # COST
    # ========================================================

    cost_100 = (
        fuel_100 *
        fuel_price
    )

    cost_500 = (
        fuel_500 *
        fuel_price
    )

    cost_1000 = (
        fuel_1000 *
        fuel_price
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Prediction Results"
    )

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Fuel Consumption",
            f"{prediction:.2f} L/100 km"
        )

    with result2:

        st.metric(
            "Efficiency Rating",
            f"{emoji} {rating}"
        )

    with result3:

        st.metric(
            "Fuel / 100 km",
            f"{fuel_100:.2f} L"
        )


    # ========================================================
    # CAR SUMMARY
    # ========================================================

    st.subheader(
        "🚘 Car Summary"
    )

    summary1, summary2, summary3, summary4 = (
        st.columns(4)
    )

    with summary1:

        st.write("**Make**")
        st.write(make)

    with summary2:

        st.write("**Model**")
        st.write(model_name)

    with summary3:

        st.write("**Year**")
        st.write(year)

    with summary4:

        st.write("**Engine**")
        st.write(f"{engine_size} L")


    # ========================================================
    # FUEL USAGE
    # ========================================================

    st.subheader(
        "⛽ Estimated Fuel Usage"
    )

    fuel1, fuel2, fuel3 = st.columns(3)

    with fuel1:

        st.metric(
            "100 km",
            f"{fuel_100:.2f} L"
        )

    with fuel2:

        st.metric(
            "500 km",
            f"{fuel_500:.2f} L"
        )

    with fuel3:

        st.metric(
            "1000 km",
            f"{fuel_1000:.2f} L"
        )


    # ========================================================
    # COST
    # ========================================================

    st.subheader(
        "💰 Estimated Fuel Cost"
    )

    cost1, cost2, cost3 = st.columns(3)

    with cost1:

        st.metric(
            "100 km",
            f"{currency_symbol} "
            f"{cost_100:,.2f}"
        )

    with cost2:

        st.metric(
            "500 km",
            f"{currency_symbol} "
            f"{cost_500:,.2f}"
        )

    with cost3:

        st.metric(
            "1000 km",
            f"{currency_symbol} "
            f"{cost_1000:,.2f}"
        )


    # ========================================================
    # EFFICIENCY MESSAGE
    # ========================================================

    if rating == "Excellent":

        st.success(
            "🌱 Excellent fuel efficiency!"
        )

    elif rating == "Good":

        st.success(
            "👍 Good fuel efficiency!"
        )

    elif rating == "Average":

        st.warning(
            "⚠️ Average fuel efficiency."
        )

    else:

        st.error(
            "🔴 Poor fuel efficiency."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚗 Car Fuel Consumption Predictor | "
    "Random Forest Regression | "
    "Machine Learning Project"
)
