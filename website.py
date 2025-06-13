# ==============================================================================
# LIBRARIES
# ==============================================================================
# Core libraries for data manipulation and file system access
import os
import numpy as np
import pandas as pd
import geopandas as gpd

# Libraries for web application and visualization
import streamlit as st
import altair as alt
import folium
from folium.plugins import BeautifyIcon

# Specialist libraries for geospatial routing
import osmnx as ox
from shapely.geometry import Point

# Streamlit component for embedding Folium maps
from streamlit_folium import st_folium

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
# Configure the Streamlit page. This must be the first Streamlit command.
# `layout="wide"` gives the dashboard a modern, full-screen feel.
st.set_page_config(
    page_title="London Burglary Analysis & Prediction",
    page_icon="🚨",
    layout="wide"
)

# ==============================================================================
# TITLE AND INTRODUCTION
# ==============================================================================
st.title("London Burglary Analysis & Prediction Dashboard")
st.markdown("An interactive dashboard to explore historical burglary data, operational plans, and future predictions across London.")

# ==============================================================================
# DATA LOADING AND CACHING
# ==============================================================================
# Use Streamlit's caching to load data only once, making the app much faster.

@st.cache_data
def load_data(csv_path, ward_shp_path, lsoa_geojson_path):
    """
    Loads and preprocesses the primary historical crime data and geographic boundaries.
    
    Args:
        csv_path (str): Path to the main crime CSV file.
        ward_shp_path (str): Path to the London Wards shapefile.
        lsoa_geojson_path (str): Path to the LSOA GeoJSON boundaries.
        
    Returns:
        tuple: A tuple containing the crime DataFrame, Ward GeoDataFrame, and LSOA GeoDataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Crime data not found at '{csv_path}'. Ensure the file is in the correct location.")
        st.stop()
    
    # --- Preprocessing Steps ---
    df.columns = df.columns.str.strip() # Sanitize column names
    df = df[df["Crime type"] == "Burglary"].copy() # Filter for Burglary only
    df['Month_dt'] = pd.to_datetime(df['Month'], format='%Y-%m') # Convert month string to datetime
    df['Year'] = df['Month_dt'].dt.year # Extract year for filtering
    
    # Load geographic files
    try:
        wards_gdf = gpd.read_file(ward_shp_path)
        lsoa_gdf = gpd.read_file(lsoa_geojson_path)
    except Exception as e:
        st.error(f"Fatal Error: A required shapefile/GeoJSON could not be read. Error: {e}")
        st.stop()
        
    # Standardize Coordinate Reference System (CRS) to WGS84 (EPSG:4326) for compatibility with Folium
    wards_gdf = wards_gdf.to_crs(epsg=4326)
    lsoa_gdf = lsoa_gdf.to_crs(epsg=4326)
    
    return df, wards_gdf, lsoa_gdf

@st.cache_data
def load_prediction_data(lsoa_pred_path, hotspot_pred_path):
    """
    Loads and preprocesses the model's prediction output files.
    
    Args:
        lsoa_pred_path (str): Path to the LSOA total predictions CSV.
        hotspot_pred_path (str): Path to the monthly future hotspot predictions CSV.
        
    Returns:
        tuple: A tuple containing the LSOA predictions, long-form monthly hotspots, and wide-form monthly hotspots.
    """
    try:
        df_lsoa_preds = pd.read_csv(lsoa_pred_path)
        df_hotspots_wide = pd.read_csv(hotspot_pred_path, index_col=0) # First column is the LSOA code index
    except FileNotFoundError as e:
        st.error(f"Fatal Error: Prediction data not found. Details: {e}")
        st.stop()
        
    # Sanitize LSOA codes to prevent merge errors due to whitespace
    df_lsoa_preds['lsoa_code'] = df_lsoa_preds['lsoa_code'].str.strip()
    df_hotspots_wide.index = df_hotspots_wide.index.str.strip()
    df_hotspots_wide.index.name = 'lsoa_code'
    
    # Convert the wide-format hotspot data (LSOA x Month) to a long format for easier plotting
    df_hotspots_long = df_hotspots_wide.reset_index().melt(
        id_vars='lsoa_code', var_name='forecast_month', value_name='predicted_incidents'
    )
    df_hotspots_long['forecast_month'] = pd.to_datetime(df_hotspots_long['forecast_month'])
    
    return df_lsoa_preds, df_hotspots_long, df_hotspots_wide

@st.cache_data
def get_road_network_route(waypoints_tuple):
    """
    Downloads road network and calculates an efficient cyclical patrol route.
    Uses the bounding box of waypoints for a robust graph and avoids relying on a single ward's polygon.
    
    Args:
        waypoints_tuple (tuple): A tuple of (longitude, latitude) tuples for hashable caching.
        
    Returns:
        list: A list of (lat, lon) coordinates representing the on-road route.
    """
    try:
        waypoints = [Point(xy) for xy in waypoints_tuple]
        
        # Create a bounding box around all waypoints with a small buffer to ensure connecting roads are included
        min_lon, max_lon = min(p.x for p in waypoints) - 0.005, max(p.x for p in waypoints) + 0.005
        min_lat, max_lat = min(p.y for p in waypoints) - 0.005, max(p.y for p in waypoints) + 0.005
        
        # Download the road network graph for the specified area
        G = ox.graph_from_bbox(max_lat, min_lat, max_lon, min_lon, network_type='drive', simplify=True)
        
        # Find the nearest road network node to each hotspot's centroid
        origin_nodes = [ox.nearest_nodes(G, point.x, point.y) for point in waypoints]
        
        # Calculate the shortest path between each consecutive pair of nodes to form the full route
        route_nodes = []
        for i in range(len(origin_nodes) - 1):
            try:
                path = ox.shortest_path(G, origin_nodes[i], origin_nodes[i+1], weight='length')
                if path:
                    route_nodes.extend(path[:-1]) # Add all but the last node to avoid duplicates
            except:
                continue # If a segment fails, skip it and continue with the rest of the route
        
        if route_nodes:
            route_nodes.append(origin_nodes[-1]) # Add the final destination node

        # Convert the list of node IDs to a list of (latitude, longitude) coordinates
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route_nodes]
        return route_coords
        
    except Exception:
        # Fallback to straight lines if the routing API fails, ensuring the app doesn't crash
        return [(p.y, p.x) for p in waypoints]

# --- Execute Data Loading ---
# Define file paths for clarity and easy modification
CSV_PATH = os.path.join("data", "final_data.csv")
WARD_SHP_PATH = os.path.join("London-wards-2018_ESRI", "London_Ward_CityMerged.shp")
LSOA_GEOJSON_PATH = "LSOA_2011_Boundaries.geojson"
df_crime, wards_gdf, lsoa_gdf = load_data(CSV_PATH, WARD_SHP_PATH, LSOA_GEOJSON_PATH)

LSOA_PRED_PATH = os.path.join("data", "lsoa_predictions_modified.csv")
HOTSPOT_PRED_PATH = os.path.join("data", "future_crime_hotspot_predictions.csv")
df_predictions, df_hotspots_long, df_hotspots_wide = load_prediction_data(LSOA_PRED_PATH, HOTSPOT_PRED_PATH)

# ==============================================================================
# DATA AGGREGATION & PREPARATION
# ==============================================================================
# This block prepares the data for the dashboard visualizations by merging and aggregating.
with st.spinner("Analyzing prediction data..."):
    # Perform a spatial join to map each LSOA to its corresponding Ward
    lsoa_with_wards_gdf = gpd.sjoin(lsoa_gdf, wards_gdf, how="inner", predicate="intersects")
    # Create a clean lookup table for LSOA code -> Ward Name
    lsoa_to_ward_map = lsoa_with_wards_gdf[['LSOA11CD', 'NAME', 'LSOA11NM']].rename(columns={'LSOA11CD': 'lsoa_code', 'NAME': 'WD24NM', 'LSOA11NM': 'lsoa_name'})
    
    # Merge prediction data with ward information
    df_predictions_with_ward = df_predictions.merge(lsoa_to_ward_map, on='lsoa_code', how='left')
    df_hotspots_with_ward = df_hotspots_long.merge(lsoa_to_ward_map, on='lsoa_code', how='left')
    
    # Clean data by removing predictions that couldn't be mapped to a ward
    df_predictions_with_ward.dropna(subset=['WD24NM'], inplace=True)
    df_hotspots_with_ward.dropna(subset=['WD24NM'], inplace=True)
    
    # Aggregate LSOA-level predictions up to the Ward level for the main map
    ward_predictions = df_predictions_with_ward.groupby('WD24NM')['predicted_burglary_count'].sum().reset_index()
    ward_predictions = ward_predictions.rename(columns={'predicted_burglary_count': 'ward_predicted_count'})
    
    # Aggregate monthly hotspot data to the Ward level for time-series charts
    ward_hotspots_forecast = df_hotspots_with_ward.groupby(['WD24NM', 'forecast_month'])['predicted_incidents'].sum().reset_index()

# ---- Police Allocation Logic ----
def calculate_police_allocation(pred_count, min_preds, max_preds, min_officers=1, max_officers=12):
    """Scales a predicted crime count to a recommended number of officers."""
    if pred_count <= 0: return min_officers
    if max_preds == min_preds or max_preds <= 0: return min_officers
    scaled_val = min_officers + ((pred_count - min_preds) / (max_preds - min_preds)) * (max_officers - min_officers)
    return int(np.round(np.clip(scaled_val, min_officers, max_officers)))

min_p, max_p = ward_predictions['ward_predicted_count'].min(), ward_predictions['ward_predicted_count'].max()
ward_predictions['police_allocation'] = ward_predictions['ward_predicted_count'].apply(lambda x: calculate_police_allocation(x, min_p, max_p))

# ==============================================================================
# SIDEBAR FILTERS
# =================================================================================================
st.sidebar.header("Filter Options")
all_years = sorted(df_crime['Year'].unique(), reverse=True)
selected_years = st.sidebar.multiselect("Select Year(s)", options=all_years, default=all_years)
all_wards = ["All Wards"] + sorted(df_crime['WD24NM'].dropna().unique())
selected_wards = st.sidebar.multiselect("Select Ward(s)", options=all_wards, default=["All Wards"])

# Apply filters to the historical dataset
df_filtered = df_crime.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
else:
    # Stop execution if no years are selected to prevent errors
    st.warning("Please select at least one year to display data.")
    st.stop()
if "All Wards" not in selected_wards and selected_wards:
    df_filtered = df_filtered[df_filtered['WD24NM'].isin(selected_wards)]
if df_filtered.empty:
    st.info("No historical burglary data found for the selected filters.")

# ==============================================================================
# MAIN PAGE LAYOUT
# ==============================================================================
# Use a single session state variable to keep track of the selected ward across tabs
# This ensures a consistent user experience.
if 'active_ward' not in st.session_state:
    st.session_state.active_ward = None

tab1, tab2 = st.tabs(["📈 Historical Data Analysis", "🚨 Model Predictions & Planning"])

# =================================================================================================
# ---- TAB 1: HISTORICAL DATA ANALYSIS ----
# =================================================================================================
with tab1:
    st.header("Geospatial and Temporal Analysis of Historical Data")

    # --- STATE 1: London-wide View (No ward selected) ---
    if st.session_state.active_ward is None:
        map_col, stats_col = st.columns([3, 2])
        with map_col:
            st.subheader("Interactive Crime Map (London Wards)")
            st.markdown("Ward boundaries are colored by historical burglary counts. **Click a ward to drill down.**")
            ward_crime_counts = df_filtered.groupby('WD24NM').size().reset_index(name='crime_count')
            map_data_gdf = wards_gdf.merge(ward_crime_counts, left_on='NAME', right_on='WD24NM', how='left').fillna(0)
            m_wards = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB dark_matter')
            choropleth_wards = folium.Choropleth(geo_data=map_data_gdf, name='choropleth', data=map_data_gdf, columns=['NAME', 'crime_count'], key_on='feature.properties.NAME', fill_color='Reds', fill_opacity=0.7, line_color='white', line_opacity=0.5, legend_name='Number of Burglaries', highlight=True).add_to(m_wards)
            folium.GeoJsonTooltip(['NAME', 'crime_count'], aliases=['Ward:', 'Burglaries:']).add_to(choropleth_wards.geojson)
            
            # Display the map and capture click events
            map_output = st_folium(m_wards, width='100%', height=500)
            if map_output and map_output.get("last_object_clicked_tooltip"):
                tooltip_text = map_output["last_object_clicked_tooltip"]
                if "Ward:" in tooltip_text:
                    st.session_state.active_ward = tooltip_text.split("Ward:")[1].split("Burglaries:")[0].strip()
                    st.rerun() # Rerun the script to switch to the drill-down view
        
        with stats_col:
            st.subheader("London-wide Overview")
            total_burglaries = df_filtered.shape[0]
            avg_per_month = total_burglaries / df_filtered['Month_dt'].dt.to_period('M').nunique() if not df_filtered.empty else 0
            with st.container(border=True):
                m1, m2 = st.columns(2)
                m1.metric(label="Total Burglaries", value=f"{total_burglaries:,}")
                m2.metric(label="Avg. Burglaries / Month", value=f"{avg_per_month:,.1f}")
            with st.container(border=True):
                st.markdown("##### 🏆 Top 10 Wards by Burglary Incidents")
                ward_crime_counts = df_filtered.groupby('WD24NM').size().reset_index(name='crime_count')
                top_10_wards = ward_crime_counts.nlargest(10, 'crime_count')
                chart = alt.Chart(top_10_wards).mark_bar().encode(x=alt.X('crime_count:Q', title='Total Incidents'), y=alt.Y('WD24NM:N', title='Ward', sort='-x'), tooltip=['WD24NM', 'crime_count']).properties(height=300).interactive()
                st.altair_chart(chart, use_container_width=True)

        st.subheader("Temporal Analysis")
        plot1_col, plot2_col = st.columns(2)
        with plot1_col, st.container(border=True):
            st.markdown("##### Monthly Trend")
            monthly_trend_all = df_filtered.groupby(df_filtered['Month_dt'].dt.to_period('M').astype(str)).size().reset_index(name='count').rename(columns={'Month_dt':'Year_Month'})
            monthly_trend_all['Month_dt'] = pd.to_datetime(monthly_trend_all['Year_Month'])
            monthly_chart = alt.Chart(monthly_trend_all).mark_line(point=True, strokeWidth=2, color='#4a90e2').encode(x=alt.X('Month_dt:T', title='Month'), y=alt.Y('count:Q', title='Number of Incidents'), tooltip=[alt.Tooltip('Month_dt:T', title='Month', format='%B %Y'), alt.Tooltip('count:Q', title='Incidents')]).interactive()
            st.altair_chart(monthly_chart, use_container_width=True)
            
        with plot2_col, st.container(border=True):
            st.markdown("##### Yearly Totals")
            yearly_trend_all = df_filtered.groupby('Year').size().reset_index(name='count')
            yearly_chart = alt.Chart(yearly_trend_all).mark_bar().encode(x=alt.X('Year:O', title='Year'), y=alt.Y('count:Q', title='Total Incidents'), tooltip=['Year', 'count']).interactive()
            st.altair_chart(yearly_chart, use_container_width=True)

    # --- STATE 2: Drill-down View (A ward has been selected) ---
    else: 
        map_col, stats_col = st.columns([3, 2])
        selected_ward_name = st.session_state.active_ward
        ward_data = df_filtered[df_filtered['WD24NM'] == selected_ward_name]
        with map_col:
            st.subheader(f"LSOA Level Drill-down for: {selected_ward_name}")
            if st.button("⬅️ Back to London View", key="hist_back"):
                st.session_state.active_ward = None
                st.rerun()
            lsoa_crime_counts = ward_data.groupby('LSOA code').size().reset_index(name='crime_count')
            ward_boundary_gdf = wards_gdf[wards_gdf['NAME'] == selected_ward_name]
            lsoas_in_ward_gdf = gpd.sjoin(lsoa_gdf, ward_boundary_gdf, how="inner", predicate="intersects")
            lsoa_map_data = lsoas_in_ward_gdf.merge(lsoa_crime_counts, left_on='LSOA11CD', right_on='LSOA code', how='left').fillna({'crime_count': 0})
            if not ward_boundary_gdf.empty:
                map_center = ward_boundary_gdf.dissolve().centroid.iloc[0]
                m_lsoa = folium.Map(location=[map_center.y, map_center.x], zoom_start=13, tiles='CartoDB dark_matter')
                choro_lsoa = folium.Choropleth(geo_data=lsoa_map_data, data=lsoa_map_data, columns=['LSOA11CD', 'crime_count'], key_on='feature.properties.LSOA11CD', fill_color='Reds', nan_fill_color='grey', fill_opacity=0.7, line_opacity=0.5, legend_name='Burglaries in LSOA', highlight=True).add_to(m_lsoa)
                folium.GeoJsonTooltip(fields=['LSOA11NM', 'crime_count'], aliases=['LSOA:', 'Burglaries:'], sticky=False).add_to(choro_lsoa.geojson)
                st_folium(m_lsoa, width='100%', height=425, returned_objects=[])
        with stats_col:
            st.subheader(f"Statistics for {selected_ward_name}")
            ward_total = ward_data.shape[0]
            with st.container(border=True):
                st.metric(label="Total Burglaries in Ward", value=f"{ward_total:,}")
            with st.container(border=True):
                st.markdown("##### 📈 Monthly Burglary Trend")
                if not ward_data.empty:
                    monthly_trend = ward_data.groupby(ward_data['Month_dt'].dt.to_period('M').astype(str)).size().reset_index(name='count').rename(columns={'Month_dt':'Year_Month'})
                    monthly_trend['Month_dt'] = pd.to_datetime(monthly_trend['Year_Month'])
                    trend_chart = alt.Chart(monthly_trend).mark_area(line={'color':'#e6550d'},color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#e6550d', offset=1)],x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Month_dt:T', title='Month'),y=alt.Y('count:Q', title='Number of Incidents', scale=alt.Scale(zero=False)),tooltip=[alt.Tooltip('Month_dt:T', title='Month', format='%B %Y'), alt.Tooltip('count:Q', title='Incidents')]).properties(height=300).interactive()
                    st.altair_chart(trend_chart, use_container_width=True)
                else:
                    st.info("No historical data to plot for this ward in the selected years.")

# =================================================================================================
# ---- TAB 2: MODEL PREDICTIONS & PLANNING ----
# =================================================================================================
with tab2:
    st.header("Ward-Level Burglary Forecast & Operational Planning")
    st.markdown("This section provides a strategic overview of predicted burglary risk and **road-based patrol routes**.")
    
    # --- STATE 1: London-wide View (No ward selected) ---
    if st.session_state.active_ward is None:
        map_col, chart_col = st.columns([3, 2])
        with map_col:
            st.subheader("Predicted Burglary Hotspots by Ward")
            st.markdown("Ward boundaries are colored by predicted burglary counts. **Click a ward to see patrol routes.**")
            ward_pred_map_gdf = wards_gdf.merge(ward_predictions, left_on='NAME', right_on='WD24NM', how='left')
            ward_pred_map_gdf['display_preds'] = ward_pred_map_gdf['ward_predicted_count'].fillna(0)
            ward_pred_map_gdf['display_police'] = ward_pred_map_gdf['police_allocation'].fillna(1)
            if not ward_pred_map_gdf.empty:
                m_ward_preds = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="CartoDB dark_matter")
                choropleth = folium.Choropleth(
                    geo_data=ward_pred_map_gdf, name='choropleth', data=ward_pred_map_gdf,
                    columns=['NAME', 'ward_predicted_count'], key_on='feature.properties.NAME',
                    fill_color='Reds', nan_fill_color='#fee5d9', # Light, "white-ish" red for no-data wards
                    fill_opacity=0.8, line_color='white', line_opacity=0.5,
                    legend_name='Predicted Burglaries', highlight=True
                ).add_to(m_ward_preds)
                folium.GeoJsonTooltip(
                    fields=['NAME', 'display_preds', 'display_police'],
                    aliases=['Ward:', 'Total Predicted Burglaries:', 'Recommended Officers (per shift):'],
                    style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
                ).add_to(choropleth.geojson)
                
                # Capture click events on the main prediction map
                map_output = st_folium(m_ward_preds, width='100%', height=600)
                if map_output and map_output.get("last_object_clicked_tooltip"):
                    tooltip_text = map_output["last_object_clicked_tooltip"]
                    if "Ward:" in tooltip_text:
                        st.session_state.active_ward = tooltip_text.split("Ward:")[1].split("Total Predicted Burglaries:")[0].strip()
                        st.rerun()
        
        with chart_col:
            with st.container(border=True):
                st.markdown("##### 🏆 Top 10 Predicted High-Risk Wards")
                if not ward_predictions.empty:
                    top_10_wards = ward_predictions.nlargest(10, 'ward_predicted_count')
                    chart = alt.Chart(top_10_wards).mark_bar(color='#e45756').encode(
                        x=alt.X('ward_predicted_count:Q', title='Total Predicted Incidents'),
                        y=alt.Y('WD24NM:N', title='Ward', sort='-x'),
                        tooltip=[alt.Tooltip('WD24NM', title='Ward'), alt.Tooltip('ward_predicted_count', title='Predicted Incidents', format='.1f')]
                    ).properties(height=300).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No ward prediction data available.")

        # Display the historical vs. predicted comparison charts
        st.divider()
        st.header("Historical vs. Predicted Trends")
        historical_yearly = df_crime.groupby('Year').size().reset_index(name='count')
        historical_yearly['Source'] = 'Historical'
        predicted_monthly = df_hotspots_long.groupby(df_hotspots_long['forecast_month'].dt.to_period('M').astype(str)).agg({'predicted_incidents':'sum'}).reset_index().rename(columns={'forecast_month':'Year_Month', 'predicted_incidents':'count'})
        predicted_monthly['Date'] = pd.to_datetime(predicted_monthly['Year_Month'])
        
        historical_monthly = df_crime.groupby(df_crime['Month_dt'].dt.to_period('M').astype(str)).size().reset_index(name='count').rename(columns={'Month_dt':'Year_Month'})
        historical_monthly['Date'] = pd.to_datetime(historical_monthly['Year_Month'])
        
        last_historical_point = historical_monthly.nlargest(1, 'Date')
        connection_df = pd.concat([last_historical_point, predicted_monthly]).sort_values('Date')
        
        chart1_col_pred, chart2_col_pred = st.columns(2)
        with chart1_col_pred, st.container(border=True):
            st.markdown("##### Yearly Totals: Historical vs. Predicted")
            predicted_yearly = predicted_monthly.copy()
            predicted_yearly['Year'] = predicted_yearly['Date'].dt.year
            predicted_yearly = predicted_yearly.groupby('Year')['count'].sum().reset_index()
            predicted_yearly['Source'] = 'Predicted'
            combined_yearly = pd.concat([historical_yearly, predicted_yearly])
            yearly_comparison_chart = alt.Chart(combined_yearly).mark_bar().encode(x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),y=alt.Y('count:Q', title='Total Incidents'),color=alt.Color('Source:N', scale=alt.Scale(domain=['Historical', 'Predicted'], range=['#4a90e2', '#e45756']), legend=alt.Legend(title="Data Source", orient="top")),tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('count:Q', title='Total Incidents', format=',.0f'), 'Source']).interactive()
            st.altair_chart(yearly_comparison_chart, use_container_width=True)
        with chart2_col_pred, st.container(border=True):
            st.markdown("##### Monthly Trend: Historical to Predicted")
            line_hist = alt.Chart(historical_monthly).mark_line(color='#4a90e2', point=alt.OverlayMarkDef(color="#4a90e2")).encode(x=alt.X('Date:T', title='Month'), y=alt.Y('count:Q', title='Number of Incidents'), tooltip=[alt.Tooltip('Date:T', format='%B %Y', title='Date'), alt.Tooltip('count:Q', title='Incidents', format=',.0f')])
            line_pred = alt.Chart(connection_df).mark_line(color='#e45756', point=alt.OverlayMarkDef(color="#e45756")).encode(x='Date:T', y='count:Q', tooltip=[alt.Tooltip('Date:T', format='%B %Y', title='Date'), alt.Tooltip('count:Q', title='Predicted Incidents', format=',.0f')])
            monthly_comparison_chart = (line_hist + line_pred).interactive()
            st.altair_chart(monthly_comparison_chart, use_container_width=True)

    # --- STATE 2: Drill-down View (A ward has been selected) ---
    else:
        map_col, chart_col = st.columns([3, 2])
        selected_ward_name = st.session_state.active_ward
        with map_col:
            st.subheader(f"LSOA Hotspots & Patrol Route for: {selected_ward_name}")
            if st.button("⬅️ Back to London View", key="pred_back"):
                st.session_state.active_ward = None
                st.rerun()

            lsoas_in_ward_df = df_predictions_with_ward[df_predictions_with_ward['WD24NM'] == selected_ward_name]
            ward_boundary_gdf = wards_gdf[wards_gdf['NAME'] == selected_ward_name]
            lsoas_in_ward_geo = gpd.sjoin(lsoa_gdf, ward_boundary_gdf, how="inner", predicate="intersects")
            lsoa_pred_map_gdf = lsoas_in_ward_geo.merge(lsoas_in_ward_df, left_on='LSOA11CD', right_on='lsoa_code', how='left').fillna({'predicted_burglary_count': 0})

            if not lsoa_pred_map_gdf.empty:
                map_center = ward_boundary_gdf.dissolve().centroid.iloc[0]
                m_lsoa_preds = folium.Map(location=[map_center.y, map_center.x], zoom_start=13, tiles="CartoDB dark_matter")
                folium.Choropleth(geo_data=lsoa_pred_map_gdf, name='lsoa_choropleth', data=lsoa_pred_map_gdf, columns=['LSOA11CD', 'predicted_burglary_count'], key_on='feature.properties.LSOA11CD', fill_color='Reds', nan_fill_color='grey', fill_opacity=0.7, line_color='white', line_opacity=0.5, legend_name='Predicted Burglaries', highlight=True).add_to(m_lsoa_preds)

                lsoa_pred_map_gdf['lsoa_code'] = lsoa_pred_map_gdf['lsoa_code'].astype(str)
                lsoa_pred_map_gdf = lsoa_pred_map_gdf.merge(df_hotspots_wide, left_on='lsoa_code', right_index=True, how='left')
                target_month = '2026-02'
                
                if target_month in lsoa_pred_map_gdf.columns:
                    high_risk_lsoas = lsoa_pred_map_gdf[lsoa_pred_map_gdf[target_month] == 1.0]
                    if len(high_risk_lsoas) >= 2:
                        with st.spinner("Calculating optimal road-based patrol route..."):
                            st.info(f"Generating cyclical patrol route for {len(high_risk_lsoas)} hotspots...")
                            unvisited_points = list(high_risk_lsoas.geometry.centroid)
                            ordered_route = []
                            current_point = min(unvisited_points, key=lambda p: p.x)
                            ordered_route.append(current_point)
                            unvisited_points.remove(current_point)
                            while unvisited_points:
                                closest_point = min(unvisited_points, key=lambda p: current_point.distance(p))
                                ordered_route.append(closest_point)
                                unvisited_points.remove(closest_point)
                                current_point = closest_point
                            ordered_route.append(ordered_route[0])
                            waypoints_tuple = tuple((p.x, p.y) for p in ordered_route)
                            route_coords = get_road_network_route(waypoints_tuple)
                            folium.PolyLine(route_coords, color='cyan', weight=4, opacity=0.8, dash_array='5, 10').add_to(m_lsoa_preds)
                    elif len(high_risk_lsoas) == 1:
                        st.info(f"Only one designated hotspot. Highlighting its location.")
                        centroid = high_risk_lsoas.geometry.centroid.iloc[0]
                        folium.Marker(location=[centroid.y, centroid.x], icon=BeautifyIcon(icon='star', background_color='#FFD700', border_color='transparent'), tooltip=f"High-Risk Hotspot: {high_risk_lsoas['lsoa_name'].iloc[0]}").add_to(m_lsoa_preds)
                    else:
                        st.info("No designated hotspots found. Displaying default perimeter patrol route.")
                        folium.GeoJson(ward_boundary_gdf.boundary, style_function=lambda x: {'color': 'gray', 'weight': 3, 'dashArray': '5, 5'}, tooltip="Default Perimeter Patrol").add_to(m_lsoa_preds)
                else:
                    st.error(f"Prediction data for target month '{target_month}' not found.")
                st_folium(m_lsoa_preds, width='100%', height=500, returned_objects=[])
        
        with chart_col:
            st.subheader(f"Forecast for {selected_ward_name}")
            ward_info_df = ward_predictions[ward_predictions['WD24NM'] == selected_ward_name]
            if not ward_info_df.empty:
                ward_info = ward_info_df.iloc[0]
            else:
                ward_info = pd.Series({'ward_predicted_count': 0, 'police_allocation': 1, 'WD24NM': selected_ward_name})
            m1, m2 = st.columns(2)
            m1.metric("Total Predicted Burglaries", f"{ward_info['ward_predicted_count']:.1f}")
            m2.metric("Recommended Officers", f"{ward_info['police_allocation']}")
            lsoas_in_ward_df = df_predictions_with_ward[df_predictions_with_ward['WD24NM'] == selected_ward_name]
            lsoa_names_in_ward = ["Entire Ward"] + sorted(lsoas_in_ward_df['lsoa_name'].unique().tolist())
            selected_forecast_level = st.selectbox("Select a forecast level:", options=lsoa_names_in_ward)
            if selected_forecast_level == "Entire Ward":
                forecast_data = ward_hotspots_forecast[ward_hotspots_forecast['WD24NM'] == selected_ward_name]
                chart_title = f"Aggregated 12-Month Forecast for {selected_ward_name}"
            else:
                # Find the lsoa_code for the selected LSOA name to filter the forecast data
                lsoa_code_to_plot = df_hotspots_with_ward[df_hotspots_with_ward['lsoa_name'] == selected_forecast_level].iloc[0]['lsoa_code']
                forecast_data = df_hotspots_long[df_hotspots_long['lsoa_code'] == lsoa_code_to_plot]
                chart_title = f"12-Month Forecast for {selected_forecast_level}"
            if not forecast_data.empty and forecast_data['predicted_incidents'].sum() > 0:
                forecast_chart = alt.Chart(forecast_data).mark_area(
                    line={'color':'darkred'}, color=alt.Gradient(
                        gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='darkred', offset=1)],
                        x1=1, x2=1, y1=1, y2=0)
                ).encode(
                    x=alt.X('forecast_month:T', title='Forecast Month', axis=alt.Axis(format='%b %Y')),
                    y=alt.Y('predicted_incidents:Q', title='Predicted Incidents', scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip('forecast_month:T', title='Month', format='%B %Y'), alt.Tooltip('predicted_incidents:Q', title='Predicted Incidents', format='.2f')]
                ).properties(title=chart_title).interactive()
                st.altair_chart(forecast_chart, use_container_width=True)
            else:
                st.warning("Forecast data not available for this selection.")