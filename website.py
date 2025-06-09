import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import folium_static
import os

# ---- Page Configuration (Set this at the top) ----
st.set_page_config(
    page_title="London Burglary Analysis",
    page_icon=" burglar_icon.png ",  # Optional: Add an icon
    layout="wide"
)

# ---- Title ----
st.title("London Burglary Analysis Dashboard")
st.markdown("An interactive dashboard to explore historical burglary data across London's boroughs and wards.")


# --- Load and preprocess data (GLOBAL FOR APP) ---
# This function is cached to improve performance. It runs only once when the data/paths change.
@st.cache_data
def load_data(csv_path, shp_path):
    """
    Loads crime data and geographical ward data, preprocesses, and merges them.
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
        wards_gdf = gpd.read_file(shp_path)
    except Exception as e:
        st.error(f"Fatal Error: Ward shapefile not found or could not be read from '{shp_path}'. Error: {e}")
        st.stop()

    # Ensure the Coordinate Reference System (CRS) is set to a web-friendly one (WGS84)
    wards_gdf = wards_gdf.to_crs(epsg=4326)

    return df, wards_gdf

# --- Execute Data Loading ---
CSV_PATH = os.path.join("data", "final_data.csv")
SHP_PATH = os.path.join("London-wards-2018_ESRI", "London_Ward_CityMerged.shp")
df_crime, wards_gdf = load_data(CSV_PATH, SHP_PATH)


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
# Use 'WD24NM' from crime data as it seems to be the ward name identifier
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

# Apply year filter
if selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]
else:
    # If nothing is selected, show a message and stop further processing for this run
    st.warning("Please select at least one year to display data.")
    st.stop()

# Apply ward filter
if "All Wards" not in selected_wards and selected_wards:
    df_filtered = df_filtered[df_filtered['WD24NM'].isin(selected_wards)]

# Handle case where the filter results in no data
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

    # --- Create Columns for Layout ---
    map_col, stats_col = st.columns([3, 2]) # Map takes 3/5 of the width, stats take 2/5

    with map_col:
        st.subheader("Interactive Crime Map")
        st.markdown("Ward boundaries are colored by burglary counts. Darker red indicates more incidents.")

        # --- Aggregate data for the map ---
        # Count crimes per ward based on the filtered data
        ward_crime_counts = df_filtered.groupby('WD24NM').size().reset_index(name='crime_count')

        # Merge crime counts with the ward geometries
        # Use a left merge to keep all wards, even those with 0 crimes in the filtered period
        map_data_gdf = wards_gdf.merge(
            ward_crime_counts,
            left_on='NAME',      # Column in shapefile
            right_on='WD24NM',   # Column in crime CSV
            how='left'
        )
        # Fill wards with no crimes (NaN) with 0
        map_data_gdf['crime_count'] = map_data_gdf['crime_count'].fillna(0)

        # --- Create the Folium Map ---
        # Center the map on London
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB dark_matter')

        # Add the choropleth layer
        choropleth = folium.Choropleth(
            geo_data=map_data_gdf,
            name='choropleth',
            data=map_data_gdf,
            columns=['NAME', 'crime_count'],
            key_on='feature.properties.NAME',
            fill_color='YlOrRd',  # Yellow-Orange-Red color scale as requested
            fill_opacity=0.7,
            line_opacity=0.4,
            legend_name='Number of Burglaries',
            highlight=True
        ).add_to(m)

        # Add a tooltip to show Ward Name and Crime Count on hover
        folium.GeoJsonTooltip(['NAME', 'crime_count'], aliases=['Ward:', 'Burglaries:']).add_to(choropleth.geojson)
        
        # Display the map
        folium_static(m, width=700, height=500)
        
        st.info(
            """
            **How to use the map:**
            - **Hover** over a ward to see its name and the total number of burglaries for the selected period.
            - **Zoom and Pan** to explore different areas.
            - **LSOA Level:** The requested drill-down to LSOA boundaries upon clicking is a complex feature. A future version could implement this by updating the map based on a ward selection to show LSOA-level data for that specific ward.
            """
        )


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
        peak_month_data = df_filtered.groupby('Year_Month').size().idxmax()
        peak_month_count = df_filtered.groupby('Year_Month').size().max()
        st.metric(label=f"Month with Most Burglaries", value=peak_month_data, delta=f"{peak_month_count} incidents", delta_color="off")


    st.divider() # Add a visual separator

    # --- Time Series Plots (Below the map and stats) ---
    st.header("Temporal Trends")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Seasonal Analysis (Monthly)")
        # Aggregate data by month for the line chart
        monthly_summary = df_filtered.groupby('Year_Month').size().reset_index(name='Burglaries')
        
        # Create Altair line chart
        seasonal_chart = alt.Chart(monthly_summary).mark_line(
            point=True, # Add points to the line
            strokeWidth=2
        ).encode(
            x=alt.X('Year_Month:T', title='Month'),
            y=alt.Y('Burglaries:Q', title='Number of Burglaries'),
            tooltip=['Year_Month:T', 'Burglaries:Q']
        ).properties(
            title='Monthly Burglary Trends'
        ).interactive() # Allows zooming and panning

        st.altair_chart(seasonal_chart, use_container_width=True)

    with plot_col2:
        st.subheader("Yearly Analysis")
        # Aggregate data by year for the bar chart
        yearly_summary = df_filtered.groupby('Year').size().reset_index(name='Burglaries')
        
        # Create Altair bar chart
        yearly_chart = alt.Chart(yearly_summary).mark_bar().encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)), # 'O' for Ordinal/Categorical
            y=alt.Y('Burglaries:Q', title='Number of Burglaries'),
            tooltip=['Year:O', 'Burglaries:Q']
        ).properties(
            title='Total Burglaries per Year'
        )

        st.altair_chart(yearly_chart, use_container_width=True)

# =================================================================================================
# ---- TAB 2: MODEL PREDICTIONS ----
# =================================================================================================
with tab2:
    st.header("Future Crime Predictions")
    st.info("This section is under development. Once the predictive model is complete, this tab will display forecasted burglary hotspots and trends.")
    st.image("https://i.imgur.com/g2n9t2H.png", caption="Placeholder for future prediction map.") # A placeholder image

# =================================================================================================
# ---- TAB 3: CROSS-ANALYSIS (Future) ----
# =================================================================================================
with tab3:
    st.header("Cross-Feature Correlation Analysis")
    st.info("This section is under development. It will feature tools like parallel coordinate plots or correlation matrices to explore relationships between different crime attributes (e.g., location, time, outcome).")
    # You can place your parallel coordinates plot here once you are ready to implement it.
    # The code from your original script would go here, adapted to use the df_filtered dataframe.