import streamlit as st
import pandas as pd
import altair as alt
import os
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from streamlit_folium import folium_static
import datetime

# ---- Title ----
st.title("Burglary Forecast Dashboard")

# --- Load and preprocess data (GLOBAL FOR APP) ---
csv_path = os.path.join("data", "final_data.csv")
try:
    @st.cache_data
    def load_and_preprocess_data(path):
        df_loaded = pd.read_csv(path)

        # Strip any whitespace from column names just in case
        df_loaded.columns = df_loaded.columns.str.strip()

        # Extract year from Month column (format:YYYY-MM)
        df_loaded["Year"] = df_loaded["Month"].str.slice(0, 4)

        # Filter for Burglary crimes only
        df_loaded = df_loaded[df_loaded["Crime type"] == "Burglary"].copy()

        # Convert 'Month' to datetime format
        df_loaded['Month_datetime'] = pd.to_datetime(df_loaded['Month'], format='%Y-%m')

        # Create a string column for year-month (e.g., "2013-01")
        df_loaded['Year_Month'] = df_loaded['Month_datetime'].dt.to_period('M').astype(str)

        # Label encode necessary columns for Parcoords
        le_lsoa = LabelEncoder()
        df_loaded['LSOA_name_encoded'] = le_lsoa.fit_transform(df_loaded['LSOA name'])
        lsoa_tickvals = sorted(df_loaded['LSOA_name_encoded'].unique())
        lsoa_ticktext = [le_lsoa.inverse_transform([val])[0] for val in lsoa_tickvals]

        le_reported_by = LabelEncoder()
        df_loaded['Reported_by_encoded'] = le_reported_by.fit_transform(df_loaded['Reported by'])
        reported_by_tickvals = sorted(df_loaded['Reported_by_encoded'].unique())
        reported_by_ticktext = [le_reported_by.inverse_transform([val])[0] for val in reported_by_tickvals]

        le_wd24nm = LabelEncoder()
        df_loaded['WD24NM_encoded'] = le_wd24nm.fit_transform(df_loaded['WD24NM'])
        wd24nm_tickvals = sorted(df_loaded['WD24NM_encoded'].unique())
        wd24nm_ticktext = [le_wd24nm.inverse_transform([val])[0] for val in wd24nm_tickvals]
        
        return df_loaded, lsoa_tickvals, lsoa_ticktext, reported_by_tickvals, reported_by_ticktext, wd24nm_tickvals, wd24nm_ticktext

    df, lsoa_tickvals, lsoa_ticktext, reported_by_tickvals, reported_by_ticktext, wd24nm_tickvals, wd24nm_ticktext = load_and_preprocess_data(csv_path)

except FileNotFoundError:
    st.error(f"File not found: {csv_path}")
    st.stop() 
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.stop()


# ---- Sidebar filter (Panel for Filter Options) ----
st.sidebar.subheader("Filter Options")

# A checkbox to show/hide the filter panel
show_filter = st.sidebar.checkbox("Show Filters", value=True)

if show_filter:
    # **Borough Selection:**
    borough_options = ["All"] + list(df["WD24NM"].dropna().unique())
    boroughs = st.sidebar.multiselect(
        "Select Boroughs", 
        options=borough_options,
        default=["All"],
        help="Choose one or more boroughs to filter the data"
    )
    
    # **Date Range Selection:**
    min_date = df['Month_datetime'].min().date()  
    max_date = df['Month_datetime'].max().date()

    date_range = st.sidebar.slider(
        "Select Date Range", 
        min_value=min_date, 
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM"
    )

    # Apply filters when "Apply Filters" button is clicked
    if st.sidebar.button("Apply Filters"):
        # Apply the borough filter
        if "All" in boroughs:
            df_filtered = df.copy()  # Do not filter by borough
        else:
            df_filtered = df[df["WD24NM"].isin(boroughs)]

        # Apply the date filter
        df_filtered = df_filtered[(df_filtered['Month_datetime'].dt.date >= date_range[0]) & 
                                   (df_filtered['Month_datetime'].dt.date <= date_range[1])]

        st.sidebar.markdown(f"**Selected Boroughs:** {', '.join(boroughs)}")
        st.sidebar.markdown(f"**Date Range:** {date_range[0]} to {date_range[1]}")
    else:
        df_filtered = df.copy()

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Historical Data", "Model Predictions", "Cross-Analysis Results"])

with tab1:
    st.subheader(f"Data for {', '.join(boroughs)}" if boroughs != ["All"] else "All London Boroughs")
    
    # Aggregate burglaries per year and month (using the filtered df)
    summary = df_filtered.groupby("Year_Month").size().reset_index(name="Burglaries")

    # ---- Plot ----
    chart = alt.Chart(summary).mark_line().encode(
        x=alt.X('Year_Month:O', title="Year-Month"),  
        y="Burglaries:Q",
        color=alt.value("blue"),
        tooltip=['Year_Month:O', 'Burglaries:Q']  
    ).properties(
        title="Monthly Burglaries with Seasonal Patterns"
    )
    
    st.altair_chart(chart, use_container_width=True)

    # ---- Interactive Map ----
    st.markdown("### Interactive Map of Crime Hotspots")

    # Create a folium map centered around London
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)  # Center map on London

    # Add HeatMap layer for burglary incidents
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df_filtered.iterrows() if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude'])]
    HeatMap(heat_data).add_to(m)

    # Add the map to Streamlit
    folium_static(m)

    # ---- Notes ----
    st.markdown("""
    *Note: This map visualizes real crime data. Final version will include model results and predicted crime hotspots.*
    """)
    
with tab2:
    st.subheader("Model Predictions")

with tab3:
    st.subheader("Cross-Analysis Results")
    
    # --- Create the Plotly Figure (using the filtered df) ---
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color='blue', showscale=False), 
            dimensions= list([  
                dict(
                    range=[df_filtered['LSOA_name_encoded'].min(), df_filtered['LSOA_name_encoded'].max()],
                    constraintrange=[df_filtered['LSOA_name_encoded'].min(), df_filtered['LSOA_name_encoded'].max()], 
                    label='Borough',
                    values=df_filtered['LSOA_name_encoded'].tolist(),
                    tickvals=lsoa_tickvals, 
                    ticktext=lsoa_ticktext
                ),
                dict(
                    range=[df_filtered['Month_numeric'].min(), df_filtered['Month_numeric'].max()],
                    constraintrange=[df_filtered['Month_numeric'].min(), df_filtered['Month_numeric'].max()], 
                    label='Month',
                    values=df_filtered['Month_numeric'].tolist()
                ),
                dict(
                    range=[df_filtered['WD24NM_encoded'].min(), df_filtered['WD24NM_encoded'].max()],
                    constraintrange=[df_filtered['WD24NM_encoded'].min(), df_filtered['WD24NM_encoded'].max()], 
                    label='Ward Name', 
                    values=df_filtered['WD24NM_encoded'].tolist(),
                    tickvals=wd24nm_tickvals, 
                    ticktext=wd24nm_ticktext
                ),
                dict(
                    range=[df_filtered['Reported_by_encoded'].min(), df_filtered['Reported_by_encoded'].max()],
                    constraintrange=[df_filtered['Reported_by_encoded'].min(), df_filtered['Reported_by_encoded'].max()], 
                    label='Reported By', 
                    values=df_filtered['Reported_by_encoded'].tolist(),
                    tickvals=reported_by_tickvals, 
                    ticktext=reported_by_ticktext
                )
            ])
        )
    )

    fig.update_layout(
        title='Crime Data Parallel Coordinates Plot'
    )

    st.plotly_chart(fig, use_container_width=True) # Use st.plotly_chart for displaying in Streamlit
