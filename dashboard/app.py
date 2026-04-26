import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(
    page_title="AirSense PK",
    page_icon="🌫️",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/processed_data.csv")
    return df

df = load_data()

st.sidebar.title("🌫️ AirSense PK")
st.sidebar.markdown("AI-Powered Air Quality Intelligence for Pakistani Cities")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Overview",
    "City Deep Dive",
    "Health Risk Prediction",
    "Pollution Map",
    "Model Insights"
])

if page == "Overview":
    st.title("🌫️ AirSense PK")
    st.markdown("### AI-Powered Air Quality Prediction and Health Risk Intelligence for Pakistani Cities")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cities Covered", "10")
    col2.metric("Total Readings", f"{len(df):,}")
    col3.metric("Date Range", "Nov 2025 - Feb 2026")
    col4.metric("Avg PM2.5", f"{df['pm2_5'].mean():.1f} μg/m³")

    st.markdown("---")

    st.subheader("Average PM2.5 by City During Smog Season")
    city_avg = df.groupby('city')['pm2_5'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(city_avg, x='city', y='pm2_5',
                 color='pm2_5',
                 color_continuous_scale='YlOrRd',
                 labels={'pm2_5': 'Avg PM2.5 (μg/m³)', 'city': 'City'},
                 title="Cities Ranked by Average PM2.5")
    fig.add_hline(y=55.5, line_dash="dash", line_color="orange",
                  annotation_text="Unhealthy threshold (55.5)")
    fig.add_hline(y=150.5, line_dash="dash", line_color="red",
                  annotation_text="Very Unhealthy threshold (150.5)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AQI Category Distribution")
    cat_counts = df['aqi_category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    fig2 = px.pie(cat_counts, values='Count', names='Category',
                  title="Distribution of Air Quality Categories")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Average PM2.5 by Hour of Day")
    hourly = df.groupby('hour')['pm2_5'].mean().reset_index()
    fig3 = px.line(hourly, x='hour', y='pm2_5',
                   title="When is Pollution Worst During the Day?",
                   labels={'pm2_5': 'Avg PM2.5 (μg/m³)', 'hour': 'Hour of Day'})
    fig3.update_traces(line_color='crimson')
    st.plotly_chart(fig3, use_container_width=True)

elif page == "City Deep Dive":
    st.title("City Deep Dive")
    st.markdown("Explore detailed air quality data for each city.")
    st.markdown("---")

    selected_city = st.selectbox("Select a City", sorted(df['city'].unique()))
    city_df = df[df['city'] == selected_city].copy()
    city_df['timestamp'] = pd.to_datetime(city_df['timestamp'])

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg PM2.5", f"{city_df['pm2_5'].mean():.1f} μg/m³")
    col2.metric("Max PM2.5", f"{city_df['pm2_5'].max():.1f} μg/m³")
    col3.metric("Most Common Risk", city_df['aqi_category'].mode()[0])

    st.markdown("---")

    st.subheader(f"PM2.5 Trend in {selected_city}")
    daily = city_df.resample('D', on='timestamp')['pm2_5'].mean().reset_index()
    fig4 = px.line(daily, x='timestamp', y='pm2_5',
                   title=f"Daily Average PM2.5 in {selected_city}",
                   labels={'pm2_5': 'PM2.5 (μg/m³)', 'timestamp': 'Date'})
    fig4.add_hline(y=55.5, line_dash="dash", line_color="orange",
                   annotation_text="Unhealthy threshold")
    fig4.add_hline(y=150.5, line_dash="dash", line_color="red",
                   annotation_text="Very Unhealthy threshold")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader(f"Pollution by Hour of Day in {selected_city}")
    hourly_city = city_df.groupby('hour')['pm2_5'].mean().reset_index()
    fig5 = px.line(hourly_city, x='hour', y='pm2_5',
                   title=f"Average PM2.5 by Hour in {selected_city}",
                   labels={'pm2_5': 'PM2.5 (μg/m³)', 'hour': 'Hour of Day'})
    fig5.update_traces(line_color='steelblue')
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader(f"AQI Category Breakdown in {selected_city}")
    cat_city = city_df['aqi_category'].value_counts().reset_index()
    cat_city.columns = ['Category', 'Count']
    fig6 = px.bar(cat_city, x='Category', y='Count',
                  color='Category',
                  title=f"Hours Spent in Each AQI Category in {selected_city}")
    st.plotly_chart(fig6, use_container_width=True)

elif page == "Health Risk Prediction":
    st.title("Health Risk Prediction")
    st.markdown("Enter current pollutant levels to predict the health risk category.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        pm25_val = st.slider("PM2.5 (μg/m³)", 0.0, 500.0, 80.0)
        pm10_val = st.slider("PM10 (μg/m³)", 0.0, 600.0, 100.0)
        co_val = st.slider("Carbon Monoxide (μg/m³)", 0.0, 1000.0, 200.0)
    with col2:
        temp_val = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
        humidity_val = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        wind_val = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)

    risk_colors = {
        'Good': '#00b300',
        'Moderate': '#ffaa00',
        'Unhealthy': '#ff4400',
        'Hazardous': '#990000'
    }

    risk_advice = {
        'Good': 'Air quality is satisfactory. Enjoy outdoor activities freely.',
        'Moderate': 'Acceptable air quality. Sensitive individuals should limit prolonged outdoor activity.',
        'Unhealthy': 'Everyone may experience health effects. Reduce outdoor activity. Wear a mask if going outside.',
        'Hazardous': 'Health alert. Avoid all outdoor activity. Schools and offices should consider closure. Vulnerable groups must stay indoors.'
    }

    model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    input_data = np.array([[
        pm10_val,co_val, 0, 0, 0, 0,
        temp_val, humidity_val, 0, wind_val, 0, 0,
        12, 1, 2026, 0
    ]])

    input_scaled = scaler.transform(input_data)
    predicted_risk = model.predict(input_scaled)[0]

    color = risk_colors[predicted_risk]
    advice = risk_advice[predicted_risk]

    st.markdown("### Predicted Health Risk:")
    st.markdown(f"""
        <div style='background-color:{color}; padding:25px; border-radius:12px; 
                    color:white; font-size:28px; font-weight:bold; text-align:center;'>
            {predicted_risk}
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**Health Advice:** {advice}")

elif page == "Pollution Map":
    st.title("Pollution Map")
    st.markdown("Air quality across Pakistani cities during peak smog season.")
    st.markdown("---")

    city_coords = {
        'Lahore': (31.5497, 74.3436),
        'Karachi': (24.8607, 67.0011),
        'Islamabad': (33.6844, 73.0479),
        'Rawalpindi': (33.5651, 73.0169),
        'Peshawar': (34.0151, 71.5249),
        'Multan': (30.1575, 71.5249),
        'Faisalabad': (31.4504, 73.1350),
        'Quetta': (30.1798, 66.9750),
        'Sialkot': (32.4945, 74.5229),
        'Rahim Yar Khan': (28.4212, 70.2989)
    }

    city_avg = df.groupby('city')['pm2_5'].mean().reset_index()
    city_avg['lat'] = city_avg['city'].map(lambda x: city_coords[x][0])
    city_avg['lon'] = city_avg['city'].map(lambda x: city_coords[x][1])
    city_avg['risk'] = city_avg['pm2_5'].apply(
        lambda x: 'Good' if x <= 12 else ('Moderate' if x <= 35.4 else ('Unhealthy' if x <= 150.4 else 'Hazardous'))
    )

    fig7 = px.scatter_mapbox(city_avg,
                              lat='lat', lon='lon',
                              color='pm2_5',
                              size='pm2_5',
                              hover_name='city',
                              hover_data={'pm2_5': ':.1f', 'risk': True},
                              color_continuous_scale='YlOrRd',
                              size_max=50,
                              zoom=4.5,
                              mapbox_style='open-street-map',
                              title="PM2.5 Levels Across Pakistani Cities")
    fig7.update_layout(height=600)
    st.plotly_chart(fig7, use_container_width=True)

elif page == "Model Insights":
    st.title("Model Insights")
    st.markdown("Machine learning model performance and feature analysis.")
    st.markdown("---")

    if os.path.exists("models/model_comparison.csv"):
        st.subheader("Model Comparison")
        model_df = pd.read_csv("models/model_comparison.csv")
        fig8 = px.bar(model_df, x='Model', y='Accuracy',
                      color='Accuracy',
                      color_continuous_scale='Greens',
                      title="Model Accuracy Comparison")
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Model comparison will appear here once training is complete.")

    if os.path.exists("models/feature_importance.png"):
        st.subheader("Feature Importance")
        st.image("models/feature_importance.png")
    else:
        st.info("Feature importance chart will appear here once training is complete.")

    if os.path.exists("models/shap_summary.png"):
        st.subheader("SHAP Analysis")
        st.image("models/shap_summary.png")
    else:
        st.info("SHAP analysis will appear here once training is complete.")