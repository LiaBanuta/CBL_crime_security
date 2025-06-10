import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
# Import the interactive folium component
from streamlit_folium import st_folium
import os

# ---- Page Configuration (Set this at the top) ----
st.set_page_config(
    page_title="London Burglary Analysis & Prediction",
    page_icon="🚨",
    layout="wide"
)

# ---- Title ----
st.title("London Burglary Analysis & Prediction Dashboard")
st.markdown("An interactive dashboard to explore historical burglary data and future predictions across London.")

# --- Load and preprocess data (GLOBAL FOR APP) ---
@st.cache_data
def load_data(csv_path, ward_shp_path, lsoa_geojson_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Crime data not found at '{csv_path}'.")
        st.stop()
    df.columns = df.columns.str.strip()
    df = df[df["Crime type"] == "Burglary"].copy()
    df['Month_dt'] = pd.to_datetime(df['Month'], format='%Y-%m')
    df['Year'] = df['Month_dt'].dt.year
    df['Year_Month'] = df['Month_dt'].dt.to_period('M').astype(str)
    try:
        wards_gdf = gpd.read_file(ward_shp_path)
        lsoa_gdf = gpd.read_file(lsoa_geojson_path)
    except Exception as e:
        st.error(f"Fatal Error: A required shapefile/GeoJSON could not be read. Error: {e}")
        st.stop()
    wards_gdf = wards_gdf.to_crs(epsg=4326)
    lsoa_gdf = lsoa_gdf.to_crs(epsg=4326)
    return df, wards_gdf, lsoa_gdf

@st.cache_data
def load_prediction_data(lsoa_pred_path, hotspot_pred_path):
    try:
        df_lsoa_preds = pd.read_csv(lsoa_pred_path)
        df_hotspots_wide = pd.read_csv(hotspot_pred_path, index_col=0)
    except FileNotFoundError as e:
        st.error(f"Fatal Error: Prediction data not found. Details: {e}")
        st.stop()

    df_lsoa_preds['lsoa_code'] = df_lsoa_preds['lsoa_code'].str.strip()
    df_hotspots_wide.index = df_hotspots_wide.index.str.strip()

    df_hotspots_wide.index.name = 'lsoa_code'
    df_hotspots_long = df_hotspots_wide.reset_index().melt(
        id_vars='lsoa_code', var_name='forecast_month', value_name='predicted_incidents'
    )
    df_hotspots_long['forecast_month'] = pd.to_datetime(df_hotspots_long['forecast_month'])
    return df_lsoa_preds, df_hotspots_long

# --- Execute Data Loading ---
CSV_PATH = os.path.join("data", "final_data.csv")
WARD_SHP_PATH = os.path.join("London-wards-2018_ESRI", "London_Ward_CityMerged.shp")
LSOA_GEOJSON_PATH = "LSOA_2011_Boundaries.geojson"
df_crime, wards_gdf, lsoa_gdf = load_data(CSV_PATH, WARD_SHP_PATH, LSOA_GEOJSON_PATH)

# Use the modified prediction file as specified by the user
LSOA_PRED_PATH = os.path.join("data", "lsoa_predictions_modified.csv")
HOTSPOT_PRED_PATH = os.path.join("data", "future_crime_hotspot_predictions.csv")
df_predictions, df_hotspots_long = load_prediction_data(LSOA_PRED_PATH, HOTSPOT_PRED_PATH)

# ---- (REVISED & FIXED) Process and Aggregate Predictions to Ward Level ----
with st.spinner("Spatially joining LSOAs to Wards and aggregating predictions..."):
    lsoa_with_wards_gdf = gpd.sjoin(lsoa_gdf, wards_gdf, how="inner", predicate="intersects")
    lsoa_to_ward_map = lsoa_with_wards_gdf[['LSOA11CD', 'NAME', 'LSOA11NM']].rename(columns={
        'LSOA11CD': 'lsoa_code', 'NAME': 'WD24NM', 'LSOA11NM': 'lsoa_name'
    })

    df_predictions_with_ward = df_predictions.merge(lsoa_to_ward_map, on='lsoa_code', how='left')
    df_hotspots_with_ward = df_hotspots_long.merge(lsoa_to_ward_map, on='lsoa_code', how='left')

    unmapped_predictions = df_predictions_with_ward['WD24NM'].isnull().sum()
    if unmapped_predictions > 0:
        st.warning(f"Could not map {unmapped_predictions} LSOA predictions to a ward. They will be excluded.")

    df_predictions_with_ward.dropna(subset=['WD24NM'], inplace=True)
    df_hotspots_with_ward.dropna(subset=['WD24NM'], inplace=True)

    ward_predictions = df_predictions_with_ward.groupby('WD24NM')['predicted_burglary_count'].sum().reset_index()
    ward_predictions = ward_predictions.rename(columns={'predicted_burglary_count': 'ward_predicted_count'})

    ward_hotspots_forecast = df_hotspots_with_ward.groupby(['WD24NM', 'forecast_month'])['predicted_incidents'].sum().reset_index()

# ---- Sidebar for Filters ----
st.sidebar.header("Filter Options")
all_years = sorted(df_crime['Year'].unique(), reverse=True)
selected_years = st.sidebar.multiselect(
    "Select Year(s)", options=all_years, default=all_years,
    help="Select the years you want to analyze."
)
all_wards = ["All Wards"] + sorted(df_crime['WD24NM'].dropna().unique())
selected_wards = st.sidebar.multiselect(
    "Select Ward(s)", options=all_wards, default=["All Wards"],
    help="Select one or more wards. 'All Wards' shows data for all of London."
)

# ---- Filtering Logic ----
df_filtered = df_crime.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
else:
    st.warning("Please select at least one year to display data."); st.stop()
if "All Wards" not in selected_wards and selected_wards:
    df_filtered = df_filtered[df_filtered['WD24NM'].isin(selected_wards)]
if df_filtered.empty:
    st.warning("No burglary data found for the selected filters. Please adjust your selection."); st.stop()

# ---- Main Page Content ----
tab1, tab2, tab3 = st.tabs(["📈 Historical Data Analysis", "🚨 Model Predictions", "🔗 Cross-Analysis (Future)"])


# =================================================================================================
# ---- TAB 1: HISTORICAL DATA ANALYSIS (LSOA HOVER FIXED) ----
# =================================================================================================
with tab1:
    st.header("Geospatial and Temporal Analysis")
    if 'selected_ward' not in st.session_state:
        st.session_state.selected_ward = None

    # ---- STATE 1: London-wide view ----
    if st.session_state.selected_ward is None:
        map_col, stats_col = st.columns([3, 2])
        with map_col:
            st.subheader("Interactive Crime Map (London Wards)")
            st.markdown("Ward boundaries are colored by burglary counts. **Click a ward to drill down.**")
            ward_crime_counts = df_filtered.groupby('WD24NM').size().reset_index(name='crime_count')
            map_data_gdf = wards_gdf.merge(ward_crime_counts, left_on='NAME', right_on='WD24NM', how='left').fillna(0)
            m_wards = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB dark_matter')
            choropleth_wards = folium.Choropleth(
                geo_data=map_data_gdf, name='choropleth', data=map_data_gdf,
                columns=['NAME', 'crime_count'], key_on='feature.properties.NAME',
                fill_color='YlOrRd', fill_opacity=0.7, line_color='white', line_opacity=0.5,
                legend_name='Number of Burglaries', highlight=True
            ).add_to(m_wards)
            folium.GeoJsonTooltip(['NAME', 'crime_count'], aliases=['Ward:', 'Burglaries:']).add_to(choropleth_wards.geojson)
            map_output = st_folium(m_wards, width='100%', height=500)
            if map_output and map_output.get("last_object_clicked_tooltip"):
                tooltip_text = map_output["last_object_clicked_tooltip"]
                if "Ward:" in tooltip_text:
                    st.session_state.selected_ward = tooltip_text.split("Ward:")[1].split("Burglaries:")[0].strip()
                    st.rerun()

        with stats_col:
            st.subheader("London-wide Overview")
            st.markdown(f"Displaying data for **{', '.join(map(str, selected_years))}**.")
            total_burglaries = df_filtered.shape[0]
            num_months = df_filtered['Year_Month'].nunique()
            avg_per_month = total_burglaries / num_months if num_months > 0 else 0
            with st.container(border=True):
                m1, m2 = st.columns(2)
                m1.metric(label="Total Burglaries", value=f"{total_burglaries:,}")
                m2.metric(label="Avg. Burglaries / Month", value=f"{avg_per_month:,.1f}")
            with st.container(border=True):
                st.markdown("##### 🏆 Top 10 Wards by Burglary Incidents")
                top_10_wards = ward_crime_counts.nlargest(10, 'crime_count')
                chart = alt.Chart(top_10_wards).mark_bar().encode(
                    x=alt.X('crime_count:Q', title='Total Incidents'),
                    y=alt.Y('WD24NM:N', title='Ward', sort='-x'),
                    tooltip=[alt.Tooltip('WD24NM', title='Ward'), alt.Tooltip('crime_count', title='Incidents')]
                ).properties(height=300).interactive()
                st.altair_chart(chart, use_container_width=True)

        st.subheader("Temporal Analysis")
        plot1_col, plot2_col = st.columns(2)
        with plot1_col, st.container(border=True):
            st.markdown("##### Monthly Trend")
            monthly_trend_all = df_filtered.groupby('Year_Month').size().reset_index(name='count')
            monthly_trend_all['Month_dt'] = pd.to_datetime(monthly_trend_all['Year_Month'])
            monthly_chart = alt.Chart(monthly_trend_all).mark_line(
                point=True, strokeWidth=2, color='#4a90e2'
            ).encode(
                x=alt.X('Month_dt:T', title='Month'),
                y=alt.Y('count:Q', title='Number of Incidents'),
                tooltip=[alt.Tooltip('Month_dt:T', title='Month', format='%B %Y'), alt.Tooltip('count:Q', title='Incidents')]
            ).interactive()
            st.altair_chart(monthly_chart, use_container_width=True)

        with plot2_col, st.container(border=True):
            st.markdown("##### Yearly Totals")
            yearly_trend_all = df_filtered.groupby('Year').size().reset_index(name='count')
            yearly_chart = alt.Chart(yearly_trend_all).mark_bar().encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y('count:Q', title='Total Incidents'),
                tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('count:Q', title='Total Incidents', format=',')]
            ).interactive()
            st.altair_chart(yearly_chart, use_container_width=True)

    # ---- STATE 2: Drill-down view for a selected ward ----
    else:
        map_col, stats_col = st.columns([3, 2])
        selected_ward_name = st.session_state.selected_ward
        ward_data = df_filtered[df_filtered['WD24NM'] == selected_ward_name]
        with map_col:
            st.subheader(f"LSOA Level Drill-down for: {selected_ward_name}")
            if st.button("⬅️ Back to London View"):
                st.session_state.selected_ward = None
                st.rerun()

            lsoa_crime_counts = ward_data.groupby('LSOA code').size().reset_index(name='crime_count')
            ward_boundary_gdf = wards_gdf[wards_gdf['NAME'] == selected_ward_name]
            lsoa_boundaries_in_ward = lsoa_gdf[lsoa_gdf['LSOA11CD'].isin(ward_data['LSOA code'].unique())]
            
            # ### FIX ### The line below is the corrected one.
            # We replace the overly broad .fillna(0) with a targeted fillna on the 'crime_count' column only.
            lsoa_map_data = lsoa_boundaries_in_ward.merge(lsoa_crime_counts, left_on='LSOA11CD', right_on='LSOA code', how='left').fillna({'crime_count': 0})

            map_center = ward_boundary_gdf.dissolve().centroid.iloc[0]
            m_lsoa = folium.Map(location=[map_center.y, map_center.x], zoom_start=13, tiles='CartoDB dark_matter')

            choro_lsoa = folium.Choropleth(
                geo_data=lsoa_map_data, data=lsoa_map_data,
                columns=['LSOA11CD', 'crime_count'], key_on='feature.properties.LSOA11CD',
                fill_color='Reds', fill_opacity=0.7, line_opacity=0.5,
                legend_name='Burglaries in LSOA', highlight=True
            ).add_to(m_lsoa)

            folium.GeoJsonTooltip(
                fields=['LSOA11NM', 'crime_count'],
                aliases=['LSOA:', 'Burglaries:'],
                sticky=False
            ).add_to(choro_lsoa.geojson)

            st_folium(m_lsoa, width='100%', height=425)

        with stats_col:
            st.subheader(f"Statistics for {selected_ward_name}")
            st.markdown(f"Displaying data for **{', '.join(map(str, selected_years))}**.")
            ward_total = ward_data.shape[0]
            with st.container(border=True):
                st.metric(label="Total Burglaries in Ward", value=f"{ward_total:,}")
            with st.container(border=True):
                st.markdown("##### 📈 Monthly Burglary Trend")
                monthly_trend = ward_data.groupby('Year_Month').size().reset_index(name='count')
                monthly_trend['Month_dt'] = pd.to_datetime(monthly_trend['Year_Month'])
                trend_chart = alt.Chart(monthly_trend).mark_area(
                    line={'color':'#e6550d'},
                    color=alt.Gradient(
                        gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#e6550d', offset=1)],
                        x1=1, x2=1, y1=1, y2=0)
                ).encode(
                    x=alt.X('Month_dt:T', title='Month'),
                    y=alt.Y('count:Q', title='Number of Incidents', scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip('Month_dt:T', title='Month', format='%B %Y'), alt.Tooltip('count:Q', title='Incidents')]
                ).properties(height=300).interactive()
                st.altair_chart(trend_chart, use_container_width=True)

# =================================================================================================
# ---- TAB 2: MODEL PREDICTIONS (NEW CHARTS ADDED) ----
# =================================================================================================
with tab2:
    st.header("Ward-Level Burglary Forecast")
    st.markdown("""
    This section provides a strategic overview of predicted burglary risk at the **Ward level**.
    - The map shows total predicted burglaries for each Ward over the next 12 months.
    - **Click a Ward** to drill down into its LSOA-level hotspots and monthly forecasts.
    """)

    if 'predicted_ward' not in st.session_state:
        st.session_state.predicted_ward = None

    map_col, chart_col = st.columns([3, 2])

    if st.session_state.predicted_ward is None:
        with map_col:
            st.subheader("Predicted Burglary Hotspots by Ward")
            ward_pred_map_gdf = wards_gdf.merge(ward_predictions, left_on='NAME', right_on='WD24NM', how='left').fillna(0)
            if not ward_pred_map_gdf.empty and ward_pred_map_gdf['ward_predicted_count'].sum() > 0:
                bins = list(ward_pred_map_gdf['ward_predicted_count'].quantile([0, 0.25, 0.5, 0.75, 0.9, 1]))
                m_ward_preds = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="CartoDB positron")
                choropleth = folium.Choropleth(
                    geo_data=ward_pred_map_gdf, name='choropleth', data=ward_pred_map_gdf,
                    columns=['NAME', 'ward_predicted_count'], key_on='feature.properties.NAME',
                    fill_color='YlOrRd', fill_opacity=0.8, line_opacity=0.3,
                    legend_name='Predicted Burglaries (Next 12 Months)', bins=bins, highlight=True
                ).add_to(m_ward_preds)
                folium.GeoJsonTooltip(fields=['NAME', 'ward_predicted_count'], aliases=['Ward:', 'Total Predicted Burglaries:'], style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;").add_to(choropleth.geojson)
                map_output = st_folium(m_ward_preds, width='100%', height=600)
                if map_output and map_output.get("last_object_clicked_tooltip"):
                    tooltip_text = map_output["last_object_clicked_tooltip"]
                    if "Ward:" in tooltip_text:
                        st.session_state.predicted_ward = tooltip_text.split("Ward:")[1].split("Total Predicted Burglaries:")[0].strip()
                        st.rerun()
            else:
                st.error("No valid prediction data to display on the map. Please check the prediction files and their LSOA codes.")
        with chart_col:
            st.subheader("Top 10 Predicted High-Risk Wards")
            st.markdown("Wards with the highest forecasted burglary counts over the next 12 months.")
            if not ward_predictions.empty:
                top_10_wards = ward_predictions.nlargest(10, 'ward_predicted_count').reset_index(drop=True)
                top_10_wards.index += 1
                st.dataframe(top_10_wards.style.format({'ward_predicted_count': '{:.1f}'}), use_container_width=True)
            else:
                st.warning("No ward prediction data available.")
    else:
        selected_ward = st.session_state.predicted_ward
        with map_col:
            st.subheader(f"LSOA Hotspot Predictions for: {selected_ward}")
            if st.button("⬅️ Back to London Ward View"):
                st.session_state.predicted_ward = None
                st.rerun()
            lsoas_in_ward_df = df_predictions_with_ward[df_predictions_with_ward['WD24NM'] == selected_ward]
            lsoa_codes_in_ward = lsoas_in_ward_df['lsoa_code'].tolist()
            lsoa_pred_map_gdf = lsoa_gdf[lsoa_gdf['LSOA11CD'].isin(lsoa_codes_in_ward)]
            lsoa_pred_map_gdf = lsoa_pred_map_gdf.merge(lsoas_in_ward_df, left_on='LSOA11CD', right_on='lsoa_code', how='left').fillna(0)
            if not lsoa_pred_map_gdf.empty:
                map_center = lsoa_pred_map_gdf.dissolve().centroid.iloc[0]
                m_lsoa_preds = folium.Map(location=[map_center.y, map_center.x], zoom_start=13, tiles="CartoDB positron")
                choro_lsoa = folium.Choropleth(
                    geo_data=lsoa_pred_map_gdf, name='lsoa_choropleth', data=lsoa_pred_map_gdf,
                    columns=['LSOA11CD', 'predicted_burglary_count'], key_on='feature.properties.LSOA11CD',
                    fill_color='Reds', fill_opacity=0.8, line_color='white', line_opacity=0.5,
                    legend_name='Predicted Burglaries in LSOA', highlight=True
                ).add_to(m_lsoa_preds)
                folium.GeoJsonTooltip(fields=['lsoa_name', 'predicted_burglary_count'], aliases=['LSOA:', 'Predicted Burglaries:']).add_to(choro_lsoa.geojson)
                st_folium(m_lsoa_preds, width='100%', height=600, returned_objects=[])
            else:
                st.warning("No LSOA prediction data available for this ward.")
        with chart_col:
            st.subheader(f"Monthly Forecast for {selected_ward}")
            lsoas_in_ward_df = df_predictions_with_ward[df_predictions_with_ward['WD24NM'] == selected_ward]
            lsoa_names_in_ward = ["Entire Ward"] + sorted(lsoas_in_ward_df['lsoa_name'].unique().tolist())
            selected_forecast_level = st.selectbox("Select a forecast level:", options=lsoa_names_in_ward)
            if selected_forecast_level == "Entire Ward":
                forecast_data = ward_hotspots_forecast[ward_hotspots_forecast['WD24NM'] == selected_ward]
                chart_title = f"Aggregated 12-Month Forecast for {selected_ward}"
            else:
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

    # ---- NEW SECTION: Historical vs. Predicted Charts ----
    st.divider()
    st.header("Historical vs. Predicted Trends")
    st.markdown("Comparing city-wide historical data against future model predictions.")

    # --- Data Prep for Comparison Charts ---
    # Yearly data
    historical_yearly = df_crime.groupby('Year').size().reset_index(name='count')
    historical_yearly['Source'] = 'Historical'
    pred_yearly_df = df_hotspots_long.copy()
    pred_yearly_df['Year'] = pred_yearly_df['forecast_month'].dt.year
    predicted_yearly = pred_yearly_df.groupby('Year')['predicted_incidents'].sum().reset_index(name='count')
    predicted_yearly['Source'] = 'Predicted'
    combined_yearly = pd.concat([historical_yearly, predicted_yearly])

    # Monthly data
    historical_monthly = df_crime.groupby('Month_dt').size().reset_index(name='count').rename(columns={'Month_dt': 'Date'})
    predicted_monthly = df_hotspots_long.groupby('forecast_month')['predicted_incidents'].sum().reset_index(name='count').rename(columns={'forecast_month': 'Date'})
    # To connect the line, create a dataframe for the red line that starts with the last point of the blue line.
    last_historical_point = historical_monthly.nlargest(1, 'Date')
    connection_df = pd.concat([last_historical_point, predicted_monthly]).sort_values('Date')

    # --- Create and Display Charts ---
    chart1_col, chart2_col = st.columns(2)
    with chart1_col:
        with st.container(border=True):
            st.markdown("##### Yearly Totals: Historical vs. Predicted")
            yearly_comparison_chart = alt.Chart(combined_yearly).mark_bar().encode(
                x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count:Q', title='Total Incidents'),
                color=alt.Color('Source:N',
                                scale=alt.Scale(domain=['Historical', 'Predicted'], range=['#4a90e2', '#e45756']),
                                legend=alt.Legend(title="Data Source", orient="top")),
                tooltip=[alt.Tooltip('Year:O', title='Year'), alt.Tooltip('count:Q', title='Total Incidents', format=',.0f'), 'Source']
            ).interactive()
            st.altair_chart(yearly_comparison_chart, use_container_width=True)

    with chart2_col:
        with st.container(border=True):
            st.markdown("##### Monthly Trend: Historical to Predicted")
            # Chart Layer 1: Historical Data (Blue)
            line_hist = alt.Chart(historical_monthly).mark_line(
                color='#4a90e2',
                point=alt.OverlayMarkDef(color="#4a90e2")
            ).encode(
                x=alt.X('Date:T', title='Month'),
                y=alt.Y('count:Q', title='Number of Incidents'),
                tooltip=[alt.Tooltip('Date:T', format='%B %Y', title='Date'), alt.Tooltip('count:Q', title='Incidents', format=',.0f')]
            )
            # Chart Layer 2: Predicted Data (Red)
            line_pred = alt.Chart(connection_df).mark_line(
                color='#e45756',
                point=alt.OverlayMarkDef(color="#e45756")
            ).encode(
                x='Date:T',
                y='count:Q',
                tooltip=[alt.Tooltip('Date:T', format='%B %Y', title='Date'), alt.Tooltip('count:Q', title='Predicted Incidents', format=',.0f')]
            )
            # Combine layers
            monthly_comparison_chart = (line_hist + line_pred).interactive()
            st.altair_chart(monthly_comparison_chart, use_container_width=True)


# =================================================================================================
# ---- TAB 3: CROSS-ANALYSIS (Unchanged) ----
# =================================================================================================
with tab3:
    st.header("Cross-Feature Correlation Analysis")
    st.info("This section is under development. It will feature tools like parallel coordinate plots or correlation matrices to explore relationships between different crime attributes.")