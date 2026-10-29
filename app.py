"""Streamlit app. Run: streamlit run app.py   (after python train.py)"""
import datetime as dt
import joblib, numpy as np, pandas as pd, streamlit as st
import aqi_utils as U

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")

@st.cache_resource
def load():
    return joblib.load("models/aqi_model.joblib")

try:
    art = load()
except FileNotFoundError:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

COLORS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffd700",
          "Poor": "#ff9900", "Very Poor": "#ff3300", "Severe": "#990000"}

def predict(city, date, vals, lag_pm25, roll_pm25, roll_pm10):
    row = {f: 0.0 for f in art["features"]}
    row.update(vals)
    row.update({"Month": date.month, "DayOfWeek": date.weekday(),
                "DayOfYear": date.timetuple().tm_yday,
                "PM25_lag1": lag_pm25, "PM25_roll7": roll_pm25, "PM10_roll7": roll_pm10})
    row[f"City_{city}"] = 1.0
    row[f"Season_{U.season_of(date.month)}"] = 1.0
    return float(np.clip(art["model"].predict(pd.DataFrame([row])[art["features"]])[0], 0, 500))

st.title("🌫️ Air Quality Index Predictor")
st.caption(f"Model: {art['best_name']} | Test RMSE {art['metrics']['RMSE']:.1f} | "
           f"R² {art['metrics']['R2']:.3f} | Category accuracy {art['bucket_accuracy']:.0%}")

c1, c2 = st.columns(2)
city = c1.selectbox("City", art["cities"])
date = c2.date_input("Date", dt.date.today())
med = art["city_medians"].loc[city]

st.subheader("Pollutant levels (µg/m³; CO in mg/m³)")
st.caption("Defaults are this city's historical medians.")
vals, cols = {}, st.columns(3)
for i, p in enumerate(art["pollutants"]):
    vals[p] = cols[i % 3].number_input(p, min_value=0.0, value=float(med[p]), step=1.0)

with st.expander("Recent PM trend (optional)"):
    lag = st.number_input("Yesterday's PM2.5", 0.0, value=float(vals["PM2.5"]))
    r25 = st.number_input("7-day avg PM2.5", 0.0, value=float(vals["PM2.5"]))
    r10 = st.number_input("7-day avg PM10", 0.0, value=float(vals["PM10"]))

if st.button("Predict AQI", type="primary"):
    aqi = predict(city, date, vals, lag, r25, r10)
    b = U.aqi_bucket(aqi)
    st.markdown(f"<div style='background:{COLORS[b]};padding:20px;border-radius:12px;"
                f"text-align:center;color:{'black' if b in ('Good','Satisfactory','Moderate','Poor') else 'white'}'>"
                f"<h1 style='margin:0'>{aqi:.0f}</h1><h3 style='margin:0'>{b}</h3></div>",
                unsafe_allow_html=True)
    st.info(U.ADVICE[b])
