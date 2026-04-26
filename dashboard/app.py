import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="AirSense PK",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top right, #1e293b, #0f172a);
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.98);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 16px;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}

.metric-card:hover {
    border-color: #3b82f6;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.82rem;
    margin: 0 0 6px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.metric-sub {
    color: #3b82f6;
    font-size: 0.75rem;
    margin: 4px 0 0 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.risk-box {
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 20px 0;
    letter-spacing: 1px;
}

.advice-box {
    background: rgba(59, 130, 246, 0.08);
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
    font-size: 15px;
    color: #cbd5e1;
}

.city-rank-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    font-weight: 600;
    padding: 0.6rem;
    font-size: 1rem;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

h1, h2, h3 {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

def scroll_to_top():
    st.markdown("""
        <script>
            var mainDiv = window.parent.document.querySelector('section.main');
            if (mainDiv) { mainDiv.scrollTop = 0; }
        </script>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/processed_data.csv")

df = load_data()

def render_metric(label, value, sub=""):
    st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
            <p class="metric-sub">{sub}</p>
        </div>
    """, unsafe_allow_html=True)

def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

CITY_COORDS = {
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

RISK_COLORS = {
    'Good': '#10b981',
    'Moderate': '#f59e0b',
    'Unhealthy': '#ef4444',
    'Hazardous': '#7f1d1d'
}

RISK_ICONS = {
    'Good': '✅',
    'Moderate': '⚠️',
    'Unhealthy': '🚨',
    'Hazardous': '☠️'
}

RISK_ADVICE = {
    'Good': 'Air quality is satisfactory. Enjoy outdoor activities freely.',
    'Moderate': 'Acceptable air quality. Sensitive individuals should limit prolonged outdoor activity.',
    'Unhealthy': 'Everyone may experience health effects. Reduce outdoor activity. Wear a mask if going outside.',
    'Hazardous': 'Health alert. Avoid ALL outdoor activity. Schools and offices should consider closure. Vulnerable groups must stay indoors.'
}

with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:20px 0 10px;'>
            <div style='font-size:42px;'>🌫️</div>
            <div style='font-size:20px; font-weight:700; color:#f8fafc;'>AirSense PK</div>
            <div style='font-size:11px; color:#64748b; margin-top:4px;'>Air Quality Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("", [
        "📊  Overview",
        "🏙️  City Deep Dive",
        "⚠️  Health Risk",
        "🗺️  Pollution Map",
        "🧠  Model Insights"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
        <div style='font-size:11px; color:#64748b; text-align:center; padding:10px 0;'>
            📅 Nov 2025 – Feb 2026<br>
            🏙️ 10 Pakistani Cities<br>
            📊 21,840 Hourly Readings<br><br>
            Updated: {datetime.now().strftime('%d %b %Y')}
        </div>
    """, unsafe_allow_html=True)

if "Overview" in page:
    scroll_to_top()
    st.markdown("<h1 style='margin-bottom:4px;'>📊 Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:24px;'>Real-time air quality intelligence across Pakistani cities during peak smog season.</p>", unsafe_allow_html=True)

    most_polluted = df.groupby('city')['pm2_5'].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Monitored Cities", "10", "Pakistan Region")
    with c2:
        render_metric("Avg PM2.5", f"{df['pm2_5'].mean():.1f} μg/m³", "All Cities Combined")
    with c3:
        render_metric("Most Polluted", most_polluted, "Highest Avg PM2.5")
    with c4:
        render_metric("Total Readings", f"{len(df):,}", "Hourly Data Points")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        city_avg = df.groupby('city')['pm2_5'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(city_avg, x='city', y='pm2_5',
                     color='pm2_5',
                     color_continuous_scale='YlOrRd',
                     labels={'pm2_5': 'Avg PM2.5 (μg/m³)', 'city': 'City'},
                     title="Cities Ranked by Average PM2.5")
        fig.add_hline(y=55.5, line_dash="dash", line_color="#f0883e",
                      annotation_text="Unhealthy (55.5)")
        fig.add_hline(y=150.5, line_dash="dash", line_color="#ef4444",
                      annotation_text="Very Unhealthy (150.5)")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        cat_counts = df['aqi_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig2 = px.pie(cat_counts, values='Count', names='Category',
                      hole=0.55,
                      title="AQI Distribution",
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    hourly = df.groupby('hour')['pm2_5'].mean().reset_index()
    fig3 = px.line(hourly, x='hour', y='pm2_5',
                   title="Average PM2.5 by Hour of Day — When is Pollution Worst?",
                   labels={'pm2_5': 'Avg PM2.5 (μg/m³)', 'hour': 'Hour of Day'})
    fig3.update_traces(line_color='#3b82f6', line_width=2.5)
    st.plotly_chart(style_chart(fig3), use_container_width=True)

elif "City Deep Dive" in page:
    scroll_to_top()
    st.markdown("<h1>🏙️ City Deep Dive</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:20px;'>Explore detailed air quality analytics for each Pakistani city.</p>", unsafe_allow_html=True)

    selected_city = st.selectbox("Select a City", sorted(df['city'].unique()))
    city_df = df[df['city'] == selected_city].copy()
    city_df['timestamp'] = pd.to_datetime(city_df['timestamp'])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Avg PM2.5", f"{city_df['pm2_5'].mean():.1f} μg/m³", "City Average")
    with c2:
        render_metric("Max PM2.5", f"{city_df['pm2_5'].max():.1f} μg/m³", "Peak Recorded")
    with c3:
        render_metric("Min PM2.5", f"{city_df['pm2_5'].min():.1f} μg/m³", "Lowest Recorded")
    with c4:
        render_metric("Most Common Risk", city_df['aqi_category'].mode()[0], "Dominant Category")

    st.markdown("<br>", unsafe_allow_html=True)

    daily = city_df.resample('D', on='timestamp')['pm2_5'].mean().reset_index()
    fig4 = px.area(daily, x='timestamp', y='pm2_5',
                   title=f"Daily PM2.5 Trend in {selected_city}",
                   labels={'pm2_5': 'PM2.5 (μg/m³)', 'timestamp': 'Date'})
    fig4.update_traces(line_color='#3b82f6', fillcolor='rgba(59,130,246,0.12)')
    fig4.add_hline(y=55.5, line_dash="dash", line_color="#f0883e",
                   annotation_text="Unhealthy")
    fig4.add_hline(y=150.5, line_dash="dash", line_color="#ef4444",
                   annotation_text="Very Unhealthy")
    st.plotly_chart(style_chart(fig4), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        hourly_city = city_df.groupby('hour')['pm2_5'].mean().reset_index()
        fig5 = px.bar(hourly_city, x='hour', y='pm2_5',
                      title=f"PM2.5 by Hour of Day in {selected_city}",
                      labels={'pm2_5': 'PM2.5 (μg/m³)', 'hour': 'Hour'},
                      color='pm2_5', color_continuous_scale='YlOrRd')
        st.plotly_chart(style_chart(fig5), use_container_width=True)

    with c2:
        cat_city = city_df['aqi_category'].value_counts().reset_index()
        cat_city.columns = ['Category', 'Count']
        fig6 = px.pie(cat_city, values='Count', names='Category',
                      hole=0.5,
                      title=f"AQI Categories in {selected_city}")
        st.plotly_chart(style_chart(fig6), use_container_width=True)

elif "Health Risk" in page:
    scroll_to_top()
    st.markdown("<h1>⚠️ Health Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:20px;'>Adjust pollutant and weather values to get an AI-powered health risk assessment.</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🏭 Pollutant Levels**")
        pm10 = st.slider("PM10 (μg/m³)", 0.0, 600.0, 100.0)
        co = st.slider("Carbon Monoxide (μg/m³)", 0.0, 1000.0, 200.0)
        hour = st.selectbox("Hour of Day", list(range(24)), index=12)
    with c2:
        st.markdown("**🌤️ Weather Conditions**")
        temp = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Health Risk"):
        try:
            model = joblib.load("models/random_forest_model.pkl")
            scaler = joblib.load("models/scaler.pkl")

            input_data = np.array([[pm10, co, 0, 0, 0, 0,
                                    temp, humidity, 0, wind, 0, 0,
                                    hour, 1, 2026, 0]])
            predicted_risk = model.predict(scaler.transform(input_data))[0]

            color = RISK_COLORS[predicted_risk]
            icon = RISK_ICONS[predicted_risk]
            advice = RISK_ADVICE[predicted_risk]

            st.markdown(f"""
                <div class='risk-box' style='background:{color};'>
                    {icon} {predicted_risk}
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='advice-box'>
                    📋 <strong>Health Advice:</strong> {advice}
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Model error: {e}")

elif "Pollution Map" in page:
    scroll_to_top()
    st.markdown("<h1>🗺️ Pollution Map</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:20px;'>Interactive geospatial view of air quality across Pakistani cities.</p>", unsafe_allow_html=True)
    st.markdown("---")

    city_avg = df.groupby('city')['pm2_5'].mean().reset_index()
    city_avg['lat'] = city_avg['city'].map(lambda x: CITY_COORDS.get(x, (30.3, 69.3))[0])
    city_avg['lon'] = city_avg['city'].map(lambda x: CITY_COORDS.get(x, (30.3, 69.3))[1])
    city_avg['risk'] = city_avg['pm2_5'].apply(
        lambda x: 'Good' if x <= 12 else ('Moderate' if x <= 35.4 else ('Unhealthy' if x <= 150.4 else 'Hazardous'))
    )

    fig7 = px.scatter_mapbox(
        city_avg, lat='lat', lon='lon',
        size='pm2_5', color='pm2_5',
        hover_name='city',
        hover_data={'pm2_5': ':.1f', 'risk': True, 'lat': False, 'lon': False},
        color_continuous_scale='YlOrRd',
        size_max=55,
        zoom=4.5,
        height=550,
        mapbox_style='carto-darkmatter',
        title="PM2.5 Concentration Across Pakistani Cities"
    )
    fig7.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("### City Rankings")
    city_sorted = city_avg.sort_values('pm2_5', ascending=False)
    for i, row in enumerate(city_sorted.itertuples(), 1):
        color = RISK_COLORS.get(row.risk, '#333')
        st.markdown(f"""
            <div class='city-rank-card'>
                <div>
                    <span style='color:#64748b; font-size:13px;'>#{i} </span>
                    <span style='color:#f8fafc; font-weight:600;'>{row.city}</span>
                    <span style='color:#64748b; font-size:13px; margin-left:8px;'>{row.pm2_5:.1f} μg/m³</span>
                </div>
                <span style='background:{color}; padding:3px 14px; border-radius:20px; font-size:12px; color:white;'>
                    {row.risk}
                </span>
            </div>
        """, unsafe_allow_html=True)

elif "Model" in page:
    scroll_to_top()
    st.markdown("<h1>🧠 Model Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:20px;'>Machine learning model performance and feature analysis.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if os.path.exists("models/model_comparison.csv"):
        model_df = pd.read_csv("models/model_comparison.csv")

        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, row in model_df.iterrows():
            with cols[i]:
                render_metric(row['Model'], f"{row['Accuracy']*100:.2f}%", "Classification Accuracy")

        st.markdown("<br>", unsafe_allow_html=True)
        fig8 = px.bar(model_df, x='Model', y='Accuracy',
                      color='Accuracy',
                      color_continuous_scale='Blues',
                      title="Model Accuracy Comparison",
                      text=model_df['Accuracy'].apply(lambda x: f"{x*100:.2f}%"))
        fig8.update_traces(textposition='outside')
        st.plotly_chart(style_chart(fig8), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("models/feature_importance.png"):
            st.markdown("### Feature Importance")
            st.image("models/feature_importance.png", use_column_width=True)
        else:
            st.info("Feature importance will appear after model training.")

    with c2:
        if os.path.exists("models/prediction_vs_actual.png"):
            st.markdown("### PM2.5 Prediction vs Actual")
            st.image("models/prediction_vs_actual.png", use_column_width=True)
        else:
            st.info("Prediction chart will appear after model training.")

    st.markdown("---")
    if os.path.exists("models/city_clusters.png"):
        st.markdown("### City Pollution Clusters (KMeans)")
        st.image("models/city_clusters.png", use_column_width=True)