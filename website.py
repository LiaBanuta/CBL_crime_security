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
    page_title="London Burglary Analysis",
    page_icon="burglar_icon.png",  # Optional: Add an icon
    layout="wide"
)

# ---- Title ----
st.title("London Burglary Analysis Dashboard")
st.markdown("An interactive dashboard to explore historical burglary data across London's boroughs and wards.")

# --- Load and preprocess data (GLOBAL FOR APP) ---
# This function is cached to improve performance. It runs only once when the data/paths change.
@st.cache_data
def load_data(csv_path, ward_shp_path, lsoa_geojson_path):
    """
    Loads crime data, geographical ward data, and LSOA data,
    preprocesses, and merges them.
    """
    # Load crime data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Crime data not found at '{csv_path}'. Please check the file path.")
        st.stop()

    # --- Preprocess Crime Data ---
    df.columns = df.columns.str.strip()
    df = df[df["Crime type"] == "Burglary"].copy()
    
    # Convert 'Month' to datetime objects for filtering and plotting
    df['Month_dt'] = pd.to_datetime(df['Month'], format='%Y-%m')
    df['Year'] = df['Month_dt'].dt.year
    df['Year_Month'] = df['Month_dt'].dt.to_period('M').astype(str)

    # Load geographical data for London Wards
    try:
        wards_gdf = gpd.read_file(ward_shp_path)
    except Exception as e:
        st.error(f"Fatal Error: Ward shapefile not found or could not be read from '{ward_shp_path}'. Error: {e}")
        st.stop()

    # Load geographical data for London LSOAs
    try:
        lsoa_gdf = gpd.read_file(lsoa_geojson_path)
    except Exception as e:
        st.error(f"Fatal Error: LSOA GeoJSON not found or could not be read from '{lsoa_geojson_path}'. Error: {e}")
        st.stop()

    # Ensure the Coordinate Reference System (CRS) is set to a web-friendly one (WGS84)
    # This is a CRITICAL step for them to overlay correctly
    wards_gdf = wards_gdf.to_crs(epsg=4326)
    lsoa_gdf = lsoa_gdf.to_crs(epsg=4326)

    return df, wards_gdf, lsoa_gdf

# --- Execute Data Loading ---
CSV_PATH = os.path.join("data", "final_data.csv")
WARD_SHP_PATH = os.path.join("London-wards-2018_ESRI", "London_Ward_CityMerged.shp")
LSOA_GEOJSON_PATH = "LSOA_2011_Boundaries.geojson"
df_crime, wards_gdf, lsoa_gdf = load_data(CSV_PATH, WARD_SHP_PATH, LSOA_GEOJSON_PATH)


# ---- Sidebar for Filters ----
st.sidebar.header("Filter Options")

# -- Year Filter --
all_years = sorted(df_crime['Year'].unique(), reverse=True)
selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=all_years,
    default=all_years,
    help="Select the years you want to analyze."
)

# -- Ward Filter --
all_wards = ["All Wards"] + sorted(df_crime['WD24NM'].dropna().unique())
selected_wards = st.sidebar.multiselect(
    "Select Ward(s)",
    options=all_wards,
    default=["All Wards"],
    help="Select one or more wards. 'All Wards' shows data for all of London."
)

# ---- Filtering Logic ----
# Start with a copy of the original dataframe
df_filtered = df_crime.copy()

if selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
else:
    st.warning("Please select at least one year to display data.")
    st.stop()

if "All Wards" not in selected_wards and selected_wards:
    df_filtered = df_filtered[df_filtered['WD24NM'].isin(selected_wards)]

if df_filtered.empty:
    st.warning("No burglary data found for the selected filters. Please adjust your selection.")
    st.stop()

# ---- Main Page Content ----
tab1, tab2, tab3 = st.tabs(["Historical Data Analysis", "Model Predictions", "Cross-Analysis (Future)"])

# =================================================================================================
# ---- TAB 1: HISTORICAL DATA ANALYSIS ----
# =================================================================================================
with tab1:
    st.header("Geospatial and Temporal Analysis")

    # --- Initialize session state to manage drill-down ---
    if 'selected_ward' not in st.session_state:
        st.session_state.selected_ward = None

    # --- Create Columns for Layout ---
    map_col, stats_col = st.columns([3, 2])

    # --- MAIN MAP LOGIC: Show Ward or LSOA map based on state ---
    if st.session_state.selected_ward is None:
        # ---- STATE 1: Display London-wide Ward Map ----
        with map_col:
            st.subheader("Interactive Crime Map (London Wards)")
            st.markdown("Ward boundaries are colored by burglary counts. **Click a ward to drill down to the LSOA level.**")

            ward_crime_counts = df_filtered.groupby('WD24NM').size().reset_index(name='crime_count')
            map_data_gdf = wards_gdf.merge(
                ward_crime_counts, left_on='NAME', right_on='WD24NM', how='left'
            )
            map_data_gdf['crime_count'] = map_data_gdf['crime_count'].fillna(0)

            m_wards = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB dark_matter')

            choropleth_wards = folium.Choropleth(
                geo_data=map_data_gdf,
                name='choropleth',
                data=map_data_gdf,
                columns=['NAME', 'crime_count'],
                key_on='feature.properties.NAME',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_color='white',
                line_opacity=0.5,
                legend_name='Number of Burglaries',
                highlight=True
            ).add_to(m_wards)

            folium.GeoJsonTooltip(['NAME', 'crime_count'], aliases=['Ward:', 'Burglaries:']).add_to(choropleth_wards.geojson)

            map_output = st_folium(m_wards, width=700, height=500)

            # Check if a ward was clicked from its tooltip
            if map_output and map_output.get("last_object_clicked_tooltip"):
                # Extract the ward name carefully
                tooltip_text = map_output["last_object_clicked_tooltip"]
                if "Ward:" in tooltip_text:
                    clicked_ward_name = tooltip_text.split("Ward:")[1].split("Burglaries:")[0].strip()
                    st.session_state.selected_ward = clicked_ward_name
                    st.rerun()

    else:
        # ---- STATE 2: Display LSOA-level map for the selected ward ----
        selected_ward_name = st.session_state.selected_ward

        with map_col:
            st.subheader(f"LSOA Level Drill-down for: {selected_ward_name}")
            if st.button("⬅️ Back to London View"):
                st.session_state.selected_ward = None
                st.rerun()

            # --- ROBUST DRILL-DOWN LOGIC WITH ERROR HANDLING ---
            try:
                # 1. Get the geometry of the selected ward. THIS IS A COMMON FAILURE POINT.
                selected_ward_geom = wards_gdf[wards_gdf['NAME'] == selected_ward_name]

                if selected_ward_geom.empty:
                    # If no match is found, inform the user and reset.
                    st.error(f"Could not find a matching geometry for ward: '{selected_ward_name}'. "
                             f"This may be due to a data mismatch between the shapefile and crime data. "
                             "Please check for trailing spaces or different naming conventions.")
                    # Add a button to gracefully return
                    if st.button("Return to Main Map"):
                        st.session_state.selected_ward = None
                        st.rerun()
                    st.stop() # Stop further execution in this block

                # 2. Find LSOAs that intersect with the selected ward's geometry
                # Using 'predicate="intersects"' is correct for finding LSOAs within the ward
                lsoas_in_ward = gpd.sjoin(lsoa_gdf, selected_ward_geom, how='inner', predicate='intersects')

                # 3. Filter crime data for ONLY the selected ward
                ward_crimes_df = df_filtered[df_filtered['WD24NM'] == selected_ward_name]
                total_ward_crimes = len(ward_crimes_df)

                # 4. Aggregate crime counts at the LSOA level for the crimes in that ward
                # NOTE: The join keys 'LSOA code' and 'LSOA11CD' must match!
                lsoa_crime_counts = ward_crimes_df.groupby('LSOA code').size().reset_index(name='lsoa_crime_count')

                # 5. Merge LSOA geometries with their specific crime counts
                lsoa_map_data = lsoas_in_ward.merge(
                    lsoa_crime_counts, left_on='LSOA11CD', right_on='LSOA code', how='left'
                )
                lsoa_map_data['lsoa_crime_count'] = lsoa_map_data['lsoa_crime_count'].fillna(0)

                # 6. Calculate the crime ratio as requested
                if total_ward_crimes > 0:
                    lsoa_map_data['crime_ratio'] = (lsoa_map_data['lsoa_crime_count'] / total_ward_crimes).round(3)
                else:
                    lsoa_map_data['crime_ratio'] = 0

                # 7. Create the LSOA map, centering on the ward
                map_center = selected_ward_geom.geometry.centroid.iloc[0]
                m_lsoa = folium.Map(location=[map_center.y, map_center.x], zoom_start=14, tiles='CartoDB positron')

                # Add the choropleth layer for LSOAs
                folium.Choropleth(
                    geo_data=lsoa_map_data,
                    name='lsoa_choropleth',
                    data=lsoa_map_data,
                    columns=['LSOA11CD', 'lsoa_crime_count'],
                    key_on='feature.properties.LSOA11CD',
                    fill_color='YlOrRd',
                    fill_opacity=0.8,
                    line_color='white',
                    line_opacity=0.6,
                    legend_name='Number of Burglaries in LSOA',
                    highlight=True
                ).add_to(m_lsoa)
                
                # Add a tooltip layer to show LSOA details
                tooltip = folium.features.GeoJsonTooltip(
                    fields=['LSOA11NM', 'lsoa_crime_count', 'crime_ratio'],
                    aliases=['LSOA:', 'Burglaries in LSOA:', 'Proportion of Ward Total:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
                folium.GeoJson(lsoa_map_data, style_function=lambda x: {'fillOpacity':0, 'weight':0}, tooltip=tooltip).add_to(m_lsoa)


                # Display the LSOA map
                st_folium(m_lsoa, width=700, height=500, returned_objects=[])

            except Exception as e:
                st.error(f"An unexpected error occurred during drill-down: {e}")
                st.warning("This could be due to issues with the LSOA geometries or data mismatches.")
                # Add a button to gracefully return
                if st.button("Return to Main Map"):
                    st.session_state.selected_ward = None
                    st.rerun()

    # --- The rest of the tab (stats and charts) remains the same ---
    with stats_col:
        # --- Total Crimes KPI ---
        total_burglaries = df_filtered.shape[0]
        st.metric(label="Total Burglaries (Selected Period)", value=f"{total_burglaries:,}")

        # --- Top 5 Wards by Crime ---
        st.subheader("Top 5 Wards")
        top_wards = df_filtered['WD24NM'].value_counts().nlargest(5).reset_index()
        top_wards.columns = ['Ward', 'Number of Burglaries']
        st.dataframe(top_wards, use_container_width=True, hide_index=True)

        # --- Highest Crime Month ---
        st.subheader("Peak Month")
        if not df_filtered.empty:
            peak_month_data = df_filtered.groupby('Year_Month').size().idxmax()
            peak_month_count = df_filtered.groupby('Year_Month').size().max()
            st.metric(label=f"Month with Most Burglaries", value=peak_month_data, delta=f"{peak_month_count} incidents", delta_color="off")

    st.divider()

    # --- Time Series Plots (Below the map and stats) ---
    st.header("Temporal Trends")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Seasonal Analysis (Monthly)")
        monthly_summary = df_filtered.groupby('Year_Month').size().reset_index(name='Burglaries')
        
        seasonal_chart = alt.Chart(monthly_summary).mark_line(
            point=True, strokeWidth=2
        ).encode(
            x=alt.X('Year_Month:T', title='Month'),
            y=alt.Y('Burglaries:Q', title='Number of Burglaries'),
            tooltip=['Year_Month:T', 'Burglaries:Q']
        ).properties(
            title='Monthly Burglary Trends'
        ).interactive()

        st.altair_chart(seasonal_chart, use_container_width=True)

    with plot_col2:
        st.subheader("Yearly Analysis")
        yearly_summary = df_filtered.groupby('Year').size().reset_index(name='Burglaries')
        
        yearly_chart = alt.Chart(yearly_summary).mark_bar().encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Burglaries:Q', title='Number of Burglaries'),
            tooltip=['Year:O', 'Burglaries:Q']
        ).properties(
            title='Total Burglaries per Year'
        )

        st.altair_chart(yearly_chart, use_container_width=True)

# =================================================================================================
# ---- TAB 2 & 3: MODEL PREDICTIONS & CROSS-ANALYSIS (Unchanged) ----
# =================================================================================================
with tab2:
    st.header("Future Crime Predictions")
    st.info("This section is under development. Once the predictive model is complete, this tab will display forecasted burglary hotspots and trends.")
    st.image("https://i.imgur.com/g2n9t2H.png", caption="Placeholder for future prediction map.")

with tab3:
    st.header("Cross-Feature Correlation Analysis")
    st.info("This section is under development. It will feature tools like parallel coordinate plots or correlation matrices to explore relationships between different crime attributes.")