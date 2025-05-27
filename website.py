import streamlit as st
import pandas as pd
import altair as alt
import os
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ---- Title ----
st.title("Burglary Forecast Dashboard")

# --- Load and preprocess data (GLOBAL FOR APP) ---
csv_path = os.path.join("data", "final_data.csv")
try:
    # Use st.cache_data for efficient data loading in Streamlit
    @st.cache_data
    def load_and_preprocess_data(path):
        df_loaded = pd.read_csv(path)
        
        # Strip any whitespace from column names just in case
        df_loaded.columns = df_loaded.columns.str.strip()

        # Extract year from Month column (format:YYYY-MM)
        df_loaded["Year"] = df_loaded["Month"].str.slice(0, 4)

        # Filter for Burglary crimes only
        df_loaded = df_loaded[df_loaded["Crime type"] == "Burglary"].copy() # .copy() to avoid SettingWithCopyWarning

        # Convert 'Month' to numerical (e.g., total months since a baseline)
        df_loaded['Month_datetime'] = pd.to_datetime(df_loaded['Month'], format='%Y-%m')
        df_loaded['Month_numeric'] = (df_loaded['Month_datetime'].dt.year - 2011) * 12 + df_loaded['Month_datetime'].dt.month

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
        
        # Return all necessary data and encoders/mappings
        return df_loaded, lsoa_tickvals, lsoa_ticktext, reported_by_tickvals, reported_by_ticktext, wd24nm_tickvals, wd24nm_ticktext

    df, lsoa_tickvals, lsoa_ticktext, reported_by_tickvals, reported_by_ticktext, wd24nm_tickvals, wd24nm_ticktext = load_and_preprocess_data(csv_path)

except FileNotFoundError:
    st.error(f"File not found: {csv_path}")
    st.stop() # Stop execution if file is not found
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.stop()


# ---- Sidebar filter ----
# Use the original 'WD24NM' column directly here since we haven't renamed it yet for the filter selection
borough_options = ["All"] + list(df["WD24NM"].dropna().unique())
borough = st.sidebar.selectbox(
    "Select Borough (WD24NM column)",
    borough_options
)

# Apply the borough filter here once globally
if borough != "All":
    df_filtered = df[df["WD24NM"] == borough].copy()
else:
    df_filtered = df.copy() # Work with a copy to avoid modifying the original cached df


# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Historical Data", "Model Predictions", "Cross-Analysis Results"])

with tab1:
    st.subheader(f"Data for {borough}" if borough != "All" else "All London Boroughs")
    
    # Aggregate burglaries per year (using the filtered df)
    summary = df_filtered.groupby("Year").size().reset_index(name="Burglaries")

    # ---- Plot ----
    chart = alt.Chart(summary).mark_line().encode(
        x="Year:O",
        y="Burglaries:Q",
        color=alt.value("blue")
    ).properties(title="Historical Burglaries")

    st.altair_chart(chart, use_container_width=True)

    # ---- Map placeholder ----
    st.markdown("### Interactive Map (coming soon)")
    st.map(pd.DataFrame({'lat': [51.5074], 'lon': [-0.1278]}))

    # ---- Notes ----
    st.markdown("""
    *Note: This dashboard is based on raw crime records. Final version will include model results from our FNN and predictions.*
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
                    tickvals=lsoa_tickvals, # Use global tickvals/text
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
                    tickvals=wd24nm_tickvals, # Use global tickvals/text
                    ticktext=wd24nm_ticktext
                ),
                dict(
                    range=[df_filtered['Reported_by_encoded'].min(), df_filtered['Reported_by_encoded'].max()],
                    constraintrange=[df_filtered['Reported_by_encoded'].min(), df_filtered['Reported_by_encoded'].max()], 
                    label='Reported By', 
                    values=df_filtered['Reported_by_encoded'].tolist(),
                    tickvals=reported_by_tickvals, # Use global tickvals/text
                    ticktext=reported_by_ticktext
                )
            ])
        )
    )

    fig.update_layout(
        title='Crime Data Parallel Coordinates Plot'
    )

    st.plotly_chart(fig, use_container_width=True) # Use st.plotly_chart for displaying in Streamlit