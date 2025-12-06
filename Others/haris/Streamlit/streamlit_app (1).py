"""
STREAMLIT INTERACTIVE DEMO APP
================================
Live Energy Forecasting Dashboard

Installation:
pip install streamlit plotly

Run:
streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Energy Forecasting Basel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Energy+Forecasting", use_column_width=True)
    st.title("⚡ Energy Forecasting")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Live Forecast", "🔍 Anomaly Detection", "📈 Model Performance", "⚙️ Experiments"]
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", "0.99", "+5%")
    with col2:
        st.metric("RMSE", "682", "-120")
    
    st.markdown("---")
    st.info("💡 **Tipp:** Wähle verschiedene Horizonte für unterschiedliche Insights!")

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    st.title("⚡ Energy Forecasting Basel")
    st.markdown("## Machine Learning für Stromverbrauch-Prognose")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Datenpunkte",
            value="140.000+",
            delta="4 Jahre Daten"
        )
    
    with col2:
        st.metric(
            label="🎯 Best R²",
            value="0.9898",
            delta="1h Forecast"
        )
    
    with col3:
        st.metric(
            label="⚡ Features",
            value="45+",
            delta="Zeit + Wetter + Lags"
        )
    
    with col4:
        st.metric(
            label="🔮 Horizonte",
            value="4",
            delta="15min bis 24h"
        )
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Projekt-Übersicht")
        st.markdown("""
        Dieses Projekt entwickelt ein **datengetriebenes System** zur:
        
        - ⚡ **Energie-Lastprognose** (15min bis 24h voraus)
        - 🚨 **Anomalie-Erkennung** (Extremereignisse identifizieren)
        - 📊 **Peak Management** (Lastspitzen vorhersagen)
        - 💰 **Kostenoptimierung** (Beschaffung optimieren)
        
        Basierend auf:
        - 📅 Historischen Stromverbrauchsdaten (2021-2024)
        - 🌤️ Wetterdaten (Temperatur, Strahlung, etc.)
        - 📆 Kalender-Features (Wochentag, Feiertage, etc.)
        """)
    
    with col2:
        st.markdown("### 🎓 Methodik")
        st.markdown("""
        **Framework:**
        - CRISP-DM Prozess
        - Scrum Sprints
        
        **ML Approach:**
        - Random Forest
        - Gradient Boosting
        - Time Series CV
        
        **Tech Stack:**
        - Python 3.11
        - scikit-learn
        - Streamlit
        """)
    
    st.markdown("---")
    
    # Demo Plot
    st.markdown("### 📈 Beispiel: 7-Tage Forecast")
    
    # Dummy data for demo
    dates = pd.date_range(start='2024-12-25', end='2025-01-01', freq='15min')
    true_values = np.sin(np.arange(len(dates)) * 0.01) * 5000 + 35000 + np.random.randn(len(dates)) * 500
    pred_values = true_values + np.random.randn(len(dates)) * 300
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=true_values,
        mode='lines',
        name='Tatsächlicher Verbrauch',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred_values,
        mode='lines',
        name='Vorhersage',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='1h Forecast - Letzte 7 Tage',
        xaxis_title='Zeit',
        yaxis_title='Stromverbrauch (kWh)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# LIVE FORECAST PAGE
# ============================================================

elif page == "📊 Live Forecast":
    st.title("📊 Live Forecast Dashboard")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.selectbox(
            "Forecast Horizont",
            ["15 Minuten", "1 Stunde", "4 Stunden", "24 Stunden"]
        )
    
    with col2:
        model = st.selectbox(
            "Modell",
            ["Random Forest", "Gradient Boosting", "Ensemble"]
        )
    
    with col3:
        features = st.multiselect(
            "Features",
            ["Lag Features", "Zeit Features", "Wetter Features"],
            default=["Lag Features", "Zeit Features"]
        )
    
    if st.button("🔮 Forecast generieren", type="primary"):
        with st.spinner("Generiere Vorhersage..."):
            # Simulate forecast
            import time
            time.sleep(1)
            
            st.success(f"✅ {horizon} Forecast mit {model} erfolgreich erstellt!")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("RMSE", "682 kWh", "-12%")
                st.metric("MAE", "524 kWh", "-8%")
            
            with col2:
                st.metric("R² Score", "0.9898", "+0.003")
                st.metric("Verbesserung vs. Baseline", "84.5%", "+2.1%")
            
            # Forecast Plot
            dates = pd.date_range(start='2024-12-30', periods=100, freq='1H')
            forecast = np.sin(np.arange(100) * 0.1) * 3000 + 38000
            lower = forecast - 500
            upper = forecast + 500
            
            fig = go.Figure()
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=dates, y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.2)',
                name='95% Konfidenzintervall'
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=dates, y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f'{horizon} Forecast mit Konfidenzintervall',
                xaxis_title='Zeit',
                yaxis_title='Stromverbrauch (kWh)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            st.download_button(
                label="📥 Forecast als CSV herunterladen",
                data="timestamp,forecast,lower,upper\n...",  # Dummy
                file_name=f"forecast_{horizon}.csv",
                mime="text/csv"
            )

# ============================================================
# ANOMALY DETECTION PAGE
# ============================================================

elif page == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection")
    
    # Method selection
    method = st.radio(
        "Erkennungsmethode",
        ["Residual-basiert", "Isolation Forest", "Statistical (Z-Score)"],
        horizontal=True
    )
    
    if method == "Residual-basiert":
        threshold = st.slider("Fehler-Schwellwert (kWh)", 500, 5000, 2000, 100)
    elif method == "Statistical (Z-Score)":
        z_threshold = st.slider("Z-Score Schwellwert", 2.0, 4.0, 3.0, 0.1)
    
    if st.button("🔍 Anomalien erkennen", type="primary"):
        with st.spinner("Analysiere Daten..."):
            import time
            time.sleep(1)
            
            # Dummy anomalies
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1D')
            consumption = np.random.randn(len(dates)) * 3000 + 40000
            
            # Add some anomalies
            anomaly_indices = np.random.choice(len(dates), 15, replace=False)
            consumption[anomaly_indices] += np.random.choice([-1, 1], 15) * 8000
            
            is_anomaly = np.zeros(len(dates), dtype=bool)
            is_anomaly[anomaly_indices] = True
            
            # Plot
            fig = go.Figure()
            
            # Normal points
            fig.add_trace(go.Scatter(
                x=dates[~is_anomaly],
                y=consumption[~is_anomaly],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4)
            ))
            
            # Anomalies
            fig.add_trace(go.Scatter(
                x=dates[is_anomaly],
                y=consumption[is_anomaly],
                mode='markers',
                name='Anomalie',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
            fig.update_layout(
                title='Erkannte Anomalien (2024)',
                xaxis_title='Datum',
                yaxis_title='Stromverbrauch (kWh)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Erkannte Anomalien", "15")
            with col2:
                st.metric("Anomalie-Rate", "4.1%")
            with col3:
                st.metric("Grösste Abweichung", "+12.450 kWh")

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    # Comparison table
    st.markdown("### 🏆 Multi-Horizon Vergleich")
    
    data = {
        'Horizont': ['15min', '1h', '4h', '24h'],
        'RMSE': [460, 682, 2045, 2139],
        'R²': [0.9954, 0.9898, 0.9085, 0.8999],
        'MAE': [320, 524, 1456, 1523],
        'Verbesserung': ['56.1%', '84.5%', '72.9%', '60.8%']
    }
    df = pd.DataFrame(data)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df, x='Horizont', y='RMSE',
            title='RMSE by Horizon',
            color='RMSE',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df, x='Horizont', y='R²',
            title='R² Score by Horizon',
            color='R²',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance (1h Forecast)")
    
    features = ['Lag_30min', 'Lag_15min', 'Lag_1h', 'hour', 'Lag_24h', 
                'dayofweek', 'Temperatur', 'Globalstrahlung']
    importance = [0.28, 0.26, 0.18, 0.12, 0.08, 0.04, 0.03, 0.01]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(color=importance, colorscale='Viridis')
    ))
    
    fig.update_layout(
        title='Top 8 Features',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EXPERIMENTS PAGE
# ============================================================

elif page == "⚙️ Experiments":
    st.title("⚙️ Experiment Tracking")
    
    st.markdown("""
    Systematisches Testen verschiedener Konfigurationen:
    - 🔧 Feature-Kombinationen
    - 📊 Train/Test Splits
    - ⏱️ Zeitintervalle
    - 🌤️ Mit/Ohne Wetter
    """)
    
    # Experiment configuration
    with st.expander("🔬 Neues Experiment starten"):
        col1, col2 = st.columns(2)
        
        with col1:
            exp_name = st.text_input("Experiment Name", "exp_001")
            features_exp = st.multiselect(
                "Features",
                ["Zeit", "Lags", "Wetter", "Kunden"],
                default=["Zeit", "Lags"]
            )
        
        with col2:
            interval = st.selectbox("Intervall", ["15min", "30min", "1h"])
            split = st.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05)
        
        if st.button("▶️ Experiment starten"):
            with st.spinner("Führe Experiment durch..."):
                import time
                time.sleep(2)
                st.success("✅ Experiment abgeschlossen!")
    
    # Experiment history
    st.markdown("### 📊 Experiment Historie")
    
    history = {
        'ID': ['exp_001', 'exp_002', 'exp_003', 'exp_004'],
        'Name': ['Baseline', 'No Weather', '30min Interval', 'More Regularization'],
        'RMSE': [682, 891, 723, 698],
        'R²': [0.9898, 0.9651, 0.9824, 0.9876],
        'Features': ['All', 'Time+Lag', 'All', 'All'],
        'Status': ['✅ Done', '✅ Done', '✅ Done', '✅ Done']
    }
    
    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    
    # Best experiment
    st.success("🏆 Best: exp_001 (Baseline) - R²=0.9898, RMSE=682")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        ⚡ Energy Forecasting Basel | FHNW ML Projekt 2024 | 
        <a href='https://github.com/yourusername/energy-forecasting'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

#python -m streamlit run "streamlit_app (1).py",pip install streamlit plotly, cd "C:\Users\you\OneDrive\Anlagen\Desktop\ordner_name\Streamlit"