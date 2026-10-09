import streamlit as st
import json
from supabase import create_client

st.set_page_config(page_title="House Price Predictor", layout="wide")
if "supabase" not in st.session_state:
    st.session_state.supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )
supabase = st.session_state.supabase
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🏠 House Price Predictor")
    st.subheader("Log in or create an account")

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in"):
            try:
                response = supabase.auth.sign_in_with_password(
                    {"email": login_email, "password": login_password}
                )
                st.session_state.user = response.user
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Create account"):
            try:
                response = supabase.auth.sign_up(
                    {"email": signup_email, "password": signup_password}
                )
                st.success("Account created! Now go to the Log in tab.")
            except Exception as e:
                st.error(f"Sign up failed: {e}")
            

    st.stop()

with open("model_export.json") as f:
    model = json.load(f)

intercept = model["intercept"]
feature_means = model["feature_means"]
coefficients = model["coefficients"]
demo_features = model["demo_features"]

st.title("🏠 House Price Predictor")
st.sidebar.write(f"Logged in as: {st.session_state.user.email}")
if st.sidebar.button("Log out"):
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()
st.write("Adjust the sliders to estimate a house's sale price, based on a Ridge Regression model (R² = 0.9072) trained on the Ames Housing dataset.")
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Model Comparison", "🕘 My History"])
with tab1:
    st.header("Enter House Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        gr_liv_area = st.slider("Living Area (sq ft)", 500, 4000, 1500)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

    with col2:
        total_bsmt_sf = st.slider("Basement Area (sq ft)", 0, 3000, 1000)
        full_bath = st.slider("Full Bathrooms", 0, 4, 2)
        year_built = st.slider("Year Built", 1900, 2010, 1990)

    with col3:
        tot_rms = st.slider("Total Rooms Above Ground", 2, 14, 6)
        fireplaces = st.slider("Fireplaces", 0, 3, 1)
        lot_area = st.slider("Lot Area (sq ft)", 1000, 20000, 9000)
    st.header("Predicted Sale Price")

    user_values = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "TotRmsAbvGrd": tot_rms,
        "Fireplaces": fireplaces,
        "LotArea": lot_area,
   }

    prediction = intercept
    for feature in feature_means:
        value = user_values.get(feature, feature_means[feature])
        prediction += coefficients[feature] * value

    st.metric(label="Estimated Price", value=f"${prediction:,.0f}")
    if st.button("💾 Save this prediction"):
        try:
            supabase.table("predictions").insert({
                "user_id": st.session_state.user.id,
                "overall_qual": overall_qual,
                "gr_liv_area": gr_liv_area,
                "garage_cars": garage_cars,
                "total_bsmt_sf": total_bsmt_sf,
                "full_bath": full_bath,
                "year_built": year_built,
                "tot_rms": tot_rms,
                "fireplaces": fireplaces,
                "lot_area": lot_area,
                "predicted_price": round(prediction, 2),
            }).execute()
            st.success("Prediction saved!")
        except Exception as e:
            st.error(f"Could not save: {e}")
with tab2:
    st.header("How Does Ridge Compare to Other Models?")

    import pandas as pd

    results = pd.DataFrame({
        "Model": ["XGBoost (Tuned)", "Ridge Regression", "Linear Regression",
                  "Random Forest (Tuned)", "Random Forest", "XGBoost", "Decision Tree"],
        "R2": [0.9109, 0.9072, 0.9009, 0.8922, 0.8849, 0.8708, 0.7019],
        "RMSE": [21624.93, 22074.98, 22801.19, 23785.18, 24583.05, 26041.60, 39552.52],
    })

    st.bar_chart(results.set_index("Model")["R2"])
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.info("This app predicts using Ridge Regression. Tuned XGBoost scored slightly higher in testing (R² = 0.9109 vs 0.9072) but Ridge was chosen here for its simpler, fully transparent formula.")
    with tab3:
        st.header("My Saved Predictions")
        try:
            result = (
                supabase.table("predictions")
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
            rows = result.data
            if rows:
                history = pd.DataFrame(rows)
                history = history.drop(columns=["id", "user_id"])
                history["created_at"] = pd.to_datetime(history["created_at"]).dt.tz_convert("Asia/Kolkata")
                history["created_at"] = history["created_at"].dt.strftime("%d %b %Y, %I:%M %p")
                st.dataframe(history, hide_index=True, use_container_width=True)
            else:
                st.info("No saved predictions yet. Save one from the Predict Price tab.")
        except Exception as e:
            st.error(f"Could not load history: {e}")       