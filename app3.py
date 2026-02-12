import streamlit as st
import numpy as np
import pickle

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Mobile Addiction Prediction",
    page_icon="📱",
    layout="wide"
)

# ------------------ Load Model & Scaler ------------------
with open("addiction_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ------------------ App Title ------------------
st.markdown("""
<h1 style='text-align: center;'>📱 Mobile App Addiction Risk Prediction</h1>
<p style='text-align: center; font-size:18px;'>Predict addiction risk based on mobile usage behavior</p>
<hr>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("🧍 Personal Details")

age = st.sidebar.number_input("Age", 10, 25, 18)

sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
weekend_usage = st.sidebar.slider("Weekend Usage Hours", 0.0, 20.0, 7.0)

# ================= MAIN INPUT AREA =================
st.subheader("📊 Usage & Lifestyle Details")

col1, col2, col3 = st.columns(3)

with col1:
    academic_perf = st.slider("Academic Performance (1–10)", 1, 10, 6)
    exercise_hours = st.slider("Exercise Hours", 0.0, 5.0, 1.0)
    phone_checks = st.number_input("Phone Checks / Day", 0, 300, 60)

with col2:
    social_interaction = st.slider("Social Interaction (1–10)", 1, 10, 6)
    anxiety = st.slider("Anxiety Level (1–10)", 1, 10, 5)
    depression = st.slider("Depression Level (1–10)", 1, 10, 4)

with col3:
    self_esteem = st.slider("Self Esteem (1–10)", 1, 10, 6)
    screen_time_bed = st.slider("Screen Time Before Bed (hrs)", 0.0, 5.0, 1.0)
    apps_used = st.number_input("Apps Used Daily", 0, 100, 15)

# ================= SCREEN TIME BREAKDOWN =================
st.subheader("⏱️ Screen Time Breakdown (Daily)")

col4, col5, col6 = st.columns(3)

with col4:
    time_social = st.slider("Social Media (hrs)", 0.0, 10.0, 3.0)
with col5:
    time_gaming = st.slider("Gaming (hrs)", 0.0, 10.0, 2.0)
with col6:
    time_education = st.slider("Education (hrs)", 0.0, 10.0, 2.0)

# ================= AUTO CALCULATION =================
total_screen_time = time_social + time_gaming + time_education

st.info(f"📱 **Total Screen Time (Calculated): {total_screen_time} hours/day**")

# ================= PREDICTION =================
st.markdown("---")

if st.button("🔍 Predict Addiction Risk", use_container_width=True):

    # -------- VALIDATION --------
    if total_screen_time > 24:
        st.error("❌ Total screen time cannot exceed 24 hours.")
        st.stop()

    if total_screen_time > (24 - sleep_hours):
        st.error("❌ Screen time exceeds available awake hours.")
        st.stop()

    # Auto-set daily usage from breakdown (BEST PRACTICE)
    daily_usage = total_screen_time

    # -------- MODEL INPUT --------
    input_data = np.array([[  
        age,
        daily_usage,
        sleep_hours,
        academic_perf,
        social_interaction,
        exercise_hours,
        anxiety,
        depression,
        self_esteem,
        screen_time_bed,
        phone_checks,
        apps_used,
        time_social,
        time_gaming,
        time_education,
        weekend_usage
    ]])

    input_scaled = scaler.transform(input_data)
    proba = model.predict(input_scaled)[0]

    # ================= RESULT =================
    st.subheader("📌 Prediction Result")

    if proba >= 8:
        st.error("🔴 **High Addiction Risk**")
        st.progress(90)
    elif proba >= 5:
        st.warning("🟡 **Medium Addiction Risk**")
        st.progress(60)
    else:
        st.success("🟢 **Low Addiction Risk**")
        st.progress(30)

    st.info(f"🔢 **Model Score:** {proba}")

# ------------------ Footer ------------------
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px;'>🚀 ML Project | Streamlit App | bhvyy</p>
""", unsafe_allow_html=True)