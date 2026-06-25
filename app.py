import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Taap Rakshak – Urban Heat Mitigation", layout="wide")
st.title("🛡️ Taap Rakshak – Interactive Urban Heat Mitigation")

CITY_OPTIONS = ['Delhi', 'Mumbai', 'Bengaluru']
city = st.sidebar.selectbox("Select City", CITY_OPTIONS)

# ------------------ Cached data loading ------------------
@st.cache_data
def load_city_data(city):
    return pd.read_csv(f'{city}_enriched.csv')

@st.cache_resource
def load_model(city):
    return joblib.load(f'{city}_model.pkl')

df = load_city_data(city)
model = load_model(city)

# ------------------ Build map with image overlay ------------------
# Bounding box of the city
lon_min, lon_max = df['Lon'].min(), df['Lon'].max()
lat_min, lat_max = df['Lat'].min(), df['Lat'].max()
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
               tiles='CartoDB positron', control_scale=True)

# Add the pre‑rendered HSI overlay
overlay_img = f'{city}_hsi_overlay.png'
folium.raster_layers.ImageOverlay(
    image=overlay_img,
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.7,
    interactive=False,
    name='Heat Stress Index'
).add_to(m)

folium.LayerControl().add_to(m)

map_data = st_folium(m, width=1000, height=600)

# ------------------ Click handler (unchanged) ------------------
if map_data['last_clicked'] is not None:
    click_lat = map_data['last_clicked']['lat']
    click_lon = map_data['last_clicked']['lng']
    distances = (df['Lat'] - click_lat)**2 + (df['Lon'] - click_lon)**2
    nearest = df.loc[distances.idxmin()]
    st.subheader(f"📍 Selected point ({click_lat:.3f}, {click_lon:.3f})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Heat Stress Index", f"{nearest['HSI']:.2f}")
    col1.metric("LST", f"{nearest['LST_base']:.1f} °C")
    col2.metric("Recommended", nearest['best_action'].replace('_',' ').title())
    col2.metric("Cooling", f"{abs(nearest['best_delta']):.2f} °C")
    cost_map = {'cool_roof':500, 'green_roof':1500, 'tree_planting':300, 'water_body':2000, 'none':0}
    col3.metric("Cost (₹ per pixel)", cost_map[nearest['best_action']])
    st.caption("Costs per 30m×30m pixel. Cool roof ₹500, Green roof ₹1500, Tree planting ₹300, Water body ₹2000.")

# ------------------ Tabs (unchanged) ------------------
tab1, tab2, tab3 = st.tabs(["📊 Driver Analysis", "🧠 Physics Validation", "📈 Strategy Comparison"])

with tab1:
    st.subheader("What drives urban heat?")
    imp_df = pd.read_csv(f'{city}_importance.csv')
    fig, ax = plt.subplots()
    ax.barh(imp_df['Feature'][:10], imp_df['Importance'][:10], color='teal')
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    st.pyplot(fig)
    st.caption("Top features from XGBoost – built‑up density, vegetation, and albedo are key.")

with tab2:
    st.subheader("Physics‑Informed AI verification")
    st.image(f'{city}_pdp.png', use_column_width=True)
    st.markdown("""
    **Partial dependence plots** show the model has learned physically correct relationships:
    - Higher **Albedo** → lower LST ✅
    - More **vegetation (NDVI)** → cooler ✅
    - Denser **built‑up (NDBI)** → hotter ✅
    """)

with tab3:
    st.subheader("Area‑wide cooling potential (if applied uniformly)")
    strategy_df = pd.read_csv(f'{city}_strategy.csv')
    st.bar_chart(strategy_df.set_index('Strategy'), width='stretch')
    st.caption("Average temperature reduction if the intervention is applied to all eligible pixels in the city.")
