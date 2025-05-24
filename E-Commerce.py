import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
from PIL import Image
st.set_page_config(layout="wide")

# streamlit run E-Commerce.py

warnings.filterwarnings("ignore")

# Load Model


@st.cache_data
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)


XGBoost_model = load_model()


st.markdown("<h1 style='text-align: center; color: #FF5733;'>🛒 E-commerce Visitor Conversion Prediction 🚀</h1>", unsafe_allow_html=True)
st.write("<hr style='border: 2px solid #FF5733;'>", unsafe_allow_html=True)


# Define all mappings in a single dictionary
MAPPINGS = {
    "DEVICE_BROWSER": {
        "GoogleAnalytics": 5, "Chrome": 2, "Safari": 7, "Samsung Internet": 8, "Firefox": 4,
        "Edge": 3, "Android Webview": 0, "Opera": 6, "Apache-HttpClient": 1},

    "DEVICE_OPERATING_SYSTEM": {
        "iOS": 6, "Android": 1, "Windows": 5, "Macintosh": 4, "Chrome OS": 2, "Linux": 3, "(not set)": 0},

    "DEVICE_IS_MOBILE": {"Yes": 1, "No": 0},

    "DEVICE_DEVICE_CATEGORY": {"mobile": 1, "desktop": 0, "tablet": 2},

    "EARLIEST_SOURCE": {"google": 24, "mobile": 30, "(direct)": 0, "facebook": 23, "criteo": 17, "Apple": 4,
                        "coupon.ae": 16, "Pricena_AE": 8, "newsletter": 31, "yaoota": 35, "SAPHybris": 10,
                        "bing": 15, "dba50a58f5fe8dc9a1c512d1d0b21639.safeframe.googlesyndication.com": 18,
                        "m.facebook.com": 28, "SocialMedia": 11, "Sony": 12, "identity.majidalfuttaim.com": 26,
                        "4cef946d011ad1fe6e2a064145c07a78.safeframe.googlesyndication.com": 2,
                        "everysaving.ae": 22, "yahoo": 33, "Facebook": 5, "metric.picodi.net": 29,
                        "e8e6941d3021c0e371c7fc1bd2fcb99c.safeframe.googlesyndication.com": 20,
                        "etisalat": 21, "dc3936dc840327a967ff5c01953d9c2a.safeframe.googlesyndication.com": 19,
                        "gulfnews": 25, "Philips": 7, "Pricena_AE_Home": 9,
                        "be485de1f44c0701f7e64396398f6eea.safeframe.googlesyndication.com": 14,
                        "yandex.ru": 34, "t.co": 32, "ae.asaan.com": 13,
                        "72ffcfd408248b4507bb3c3176006456.safeframe.googlesyndication.com": 3,
                        "l.messenger.com": 27, "1efcec842380aaae7d937935fb038f7d.safeframe.googlesyndication.com": 1, "MyCLUBList": 6},

    "EARLIEST_MEDIUM": {"cpc": 7, "(none)": 0, "push": 12, "inapp": 9, "organic": 11, "search": 15,
                        "cartpush": 4, "coupon": 6, "email": 8, "reactivationpush": 13, "referral": 14,
                        "signupinapp": 16, "nosignuppush": 10, "WhatsApp": 2, "sms": 17, "channelsight": 5,
                        "SocialMedia": 1, "app": 3},

    "EARLIEST_ISTRUEDIRECT": {"Yes": 1, "No": 0},

    "LATEST_ISTRUEDIRECT": {"Yes": 1, "No": 0}
}


with st.sidebar:
    select = option_menu(
        "MAIN MENU",
        ["Home", "Predict Conversion", "About"],
        icons=["house", "graph-up", "info-circle"],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "icon": {"color": "#cc3366", "font-size": "20px"},
            "container": {"background-color": "#f6eabe"},
            "nav-link": {"color": "black", "font-size": "16px"},
            "nav-link-selected": {"background-color": "#4CAF50", "color": "white"}
        }
    )


if select == "Home":
    # Update with your image path
    img = Image.open(
        r"C:/Sudhakar/Projects/Guvi Final Project/Dataset and Document/E-Commerce.jpg")
    st.image(img, width=600)

    st.markdown("<h2 style='color:green;'>E-commerce Visitor Behavior:</h2>",
                unsafe_allow_html=True)

    st.write('''E-commerce platforms attract a wide range of visitors, but not all of them convert into customers.
        Understanding visitor behavior helps businesses optimize their strategies to increase conversion rates.''')

    st.markdown("<h2 style='color:green;'>Visitor Tracking:</h2>",
                unsafe_allow_html=True)

    st.write('''Websites track various visitor interactions, such as session duration, page views, device type, and click patterns.
        These metrics help identify which users are likely to convert into customers.''')

    st.markdown("<h2 style='color:green;'>Conversion Factors:</h2>",
                unsafe_allow_html=True)

    st.write('''Several factors influence visitor conversion, including product pricing, user experience, targeted ads,
        and personalized recommendations.''')

    st.markdown("<h2 style='color:green;'>Machine Learning in Conversion Prediction:</h2>",
                unsafe_allow_html=True)

    st.write("Machine learning models can analyze past visitor data to predict whether a user is likely to convert based on their behavior.")

    st.markdown("<h2 style='color:green;'>Business Impact:</h2>",
                unsafe_allow_html=True)

    st.write("Predicting visitor conversions allows businesses to personalize user experiences, optimize marketing campaigns, and improve ROI.")

    st.markdown("<h2 style='color:green;'>A/B Testing and Optimization:</h2>",
                unsafe_allow_html=True)

    st.write("By testing different strategies, businesses can refine their approach to maximize conversion rates.")

    st.markdown("<h2 style='color:green;'>Real-time Prediction:</h2>",
                unsafe_allow_html=True)

    st.write("With live data processing, businesses can take immediate actions to engage high-potential customers before they leave the website.")


elif select == "Predict Conversion":

    # st.title("E-commerce Visitor Conversion Prediction")
    st.markdown(
        "<h1 style='color:#cc3366;'>E-commerce Visitor Conversion Prediction</h1>",
        unsafe_allow_html=True
    )

    st.subheader("Enter the Session Details Below:")

    # Create two columns for better UI layout
    col1, col2 = st.columns(2)

    # Function to create dropdowns using predefined mappings
    def create_dropdown(label, mapping, col):
        selected_key = col.selectbox(label, list(mapping.keys()))
        return mapping[selected_key]

    with col1:
        count_session = st.number_input(
            "Session Count (1 - 143)", min_value=1, max_value=143, value=(1+143)//2, step=1)
        totals_newVisits = st.number_input(
            "New Visits (0 - 1)", min_value=0, max_value=1, value=0, step=1)
        device_operatingSystem = create_dropdown(
            "Operating System", MAPPINGS["DEVICE_OPERATING_SYSTEM"], col1)
        device_deviceCategory = create_dropdown(
            "Device Category", MAPPINGS["DEVICE_DEVICE_CATEGORY"], col1)
        historic_session_page = st.number_input(
            "Historic Sessions (1 - 3066)", min_value=1, max_value=3066, value=(1+3066)//2, step=10)
        avg_session_time_page = st.number_input(
            "Avg Time per Page (sec) (1 - 3698)", min_value=1, max_value=3698, value=(1+3698)//2, step=10)
        sessionQualityDim = st.number_input(
            "Session Quality Score (1 - 67)", min_value=1, max_value=67, value=(1+67)//2, step=1)
        earliest_visit_number = st.number_input(
            "Earliest Visit Number (1 - 528)", min_value=1, max_value=528, value=(1+528)//2, step=1)
        visits_per_day = st.number_input(
            "Visits Per Day (1 - 7758)", min_value=1, max_value=7758, value=(1+7758)//2, step=10)
        earliest_medium = create_dropdown(
            "Earliest Medium", MAPPINGS["EARLIEST_MEDIUM"], col2)
        latest_keyword = st.number_input(
            "Latest Keyword ID (1 - 572)", min_value=1, max_value=572, value=(1+572)//2, step=1)
        latest_isTrueDirect = create_dropdown(
            "Latest True Direct?", MAPPINGS["LATEST_ISTRUEDIRECT"], col2)
        time_on_site = st.number_input(
            "Time on Site (sec) (1 - 6002)", min_value=1, max_value=6002, value=(1+6002)//2, step=10)
        products_array = st.number_input(
            "Products Viewed (1 - 3559)", min_value=1, max_value=3559, value=(1+3559)//2, step=1)

    with col2:
        count_hit = st.number_input(
            "Total Hits (1 - 4474)", min_value=1, max_value=4474, value=(1+4474)//2, step=10)
        device_browser = create_dropdown(
            "Device Browser", MAPPINGS["DEVICE_BROWSER"], col1)
        device_isMobile = create_dropdown(
            "Is Mobile?", MAPPINGS["DEVICE_IS_MOBILE"], col1)
        geoNetwork_region = st.number_input(
            "Region ID (1 - 91)", min_value=1, max_value=91, value=(1+91)//2, step=1)
        avg_session_time = st.number_input(
            "Avg Session Time (sec) (1 - 8068)", min_value=1, max_value=8068, value=(1+8068)//2, step=10)
        single_page_rate = st.number_input(
            "Single Page Rate (1 - 344)", min_value=1, max_value=344, value=(1+344)//2, step=1)
        earliest_visit_id = st.number_input(
            "Earliest Visit ID (1 - 7639)", min_value=1, max_value=7639, value=(1+7639)//2, step=100)
        days_since_first_visit = st.number_input(
            "Days Since First Visit (1 - 31)", min_value=1, max_value=31, value=(1+31)//2, step=1)
        earliest_source = create_dropdown(
            "Earliest Source", MAPPINGS["EARLIEST_SOURCE"], col2)
        earliest_keyword = st.number_input(
            "Earliest Keyword ID (1 - 410)", min_value=1, max_value=410, value=(1+410)//2, step=1)
        earliest_isTrueDirect = create_dropdown(
            "Earliest True Direct?", MAPPINGS["EARLIEST_ISTRUEDIRECT"], col2)
        num_interactions = st.number_input(
            "Total Interactions (1 - 6851)", min_value=1, max_value=6851, value=(1+6851)//2, step=10)
        transactionRevenue = st.number_input(
            "Transaction Revenue (1 - 4755)", min_value=1, max_value=4755, value=(1+4755)//2, step=10)

    # Predict Button
    if st.button("Predict Conversion"):
        user_data = np.array([[count_session, count_hit, totals_newVisits, device_browser, device_operatingSystem, device_isMobile,
                               device_deviceCategory, geoNetwork_region, historic_session_page, avg_session_time, avg_session_time_page,
                               single_page_rate, sessionQualityDim, earliest_visit_id, earliest_visit_number, days_since_first_visit,
                               visits_per_day, earliest_source, earliest_medium, earliest_keyword, latest_keyword, earliest_isTrueDirect,
                               latest_isTrueDirect, num_interactions, time_on_site, transactionRevenue, products_array]])

        # Display selected user inputs
        st.write("### User Input Summary:")
        st.write(f"- **Click Count (Total Hits)**: {count_hit}")
        st.write(f"- **Session Count**: {count_session}")
        st.write(
            f"- **Device**:  {'Mobile' if device_deviceCategory == 1 else 'Tablet' if device_deviceCategory == 2 else 'Desktop'}")

        # Predict using trained model
        prediction = XGBoost_model.predict(user_data)
        # Display Prediction Result
        st.success(
            f"### Prediction: {'🟢 Converted' if prediction >= 0.5 else '🔴 Not Converted'}")


elif select == "About":

    st.header(":blue[Data Collection and Preprocessing:]")
    st.write("Collect a dataset of e-commerce visitor interactions, including session details, click patterns, and device information. Preprocess the data to clean and structure it for machine learning.")

    st.header(":blue[Feature Engineering:]")
    st.write("Extract relevant features from the dataset, including session count, device type, time spent on the website, and click interactions. Create additional features that may enhance prediction accuracy.")

    st.header(":blue[Model Selection and Training:]")
    st.write("Choose an appropriate machine learning model for classification (e.g., logistic regression, decision trees, random forests, or neural networks). Train the model on historical data using a portion of the dataset for training.")

    st.header(":blue[Model Evaluation:]")
    st.write("Evaluate the model's predictive performance using classification metrics such as Precision, Recall, Accuracy, and F1-score.")

    st.header(":blue[Streamlit Web Application:]")
    st.write("Develop a user-friendly web application using Streamlit that allows users to input visitor details (click count, session count, device type, etc.). Utilize the trained machine learning model to predict whether a visitor will convert or not.")

    st.header(":blue[Deployment on Render:]")
    st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")

    st.header(":blue[Testing and Validation:]")
    st.write("Thoroughly test the deployed application to ensure it functions correctly and provides accurate conversion predictions.")
