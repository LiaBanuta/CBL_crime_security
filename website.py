import streamlit as st
import pandas as pd
import altair as alt
import os

# ---- Title ----
st.title("Burglary Forecast Dashboard")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Historical Data", "Model Predictions", "Cross-Validation Results"])

# ---- Sidebar filter ----
borough = st.sidebar.selectbox(
    "Select Borough (WD24NM column)",
    ["All"] + list(pd.read_csv("data/final_data.csv")["WD24NM"].dropna().unique())
)
with tab1:
    # ---- Load and preprocess data ----
    csv_path = os.path.join("data", "final_data.csv")
    try:
        df = pd.read_csv(csv_path)

        # Extract year from Month column (format: YYYY-MM)
        df["Year"] = df["Month"].str.slice(0, 4)

        # Filter for Burglary crimes only
        df = df[df["Crime type"] == "Burglary"]

        # Choose borough column
        df = df.rename(columns={"WD24NM": "Borough"})

        # Apply filter
        if borough != "All":
            df = df[df["Borough"] == borough]
            st.subheader(f"Data for {borough}")
        else:
            st.subheader("All London Boroughs")

        # Aggregate burglaries per year
        summary = df.groupby("Year").size().reset_index(name="Burglaries")

        # ---- Plot ----
        chart = alt.Chart(summary).mark_line().encode(
            x="Year:O",
            y="Burglaries:Q",
            color=alt.value("blue")
        ).properties(title="Historical Burglaries")

        st.altair_chart(chart, use_container_width=True)

    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")

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
    st.subheader("Cross-Validation Results")