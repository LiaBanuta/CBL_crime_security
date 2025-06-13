# Data-Driven Police Resource Allocation for Burglary Reduction in London

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive web dashboard for analyzing historical burglary data and providing operational recommendations for the Metropolitan Police Service, including predictive hotspots, officer allocation, and optimized patrol routes.

## Project Abstract

Police forces are increasingly tasked with broader responsibilities while operating under constrained resources. In London, residential burglary remains a major concern, with an unsolved rate of approximately 95.6%. This project addresses this challenge by proposing a data-driven, automated demand forecasting system to support strategic police resource allocation.

The central research question is:
> *How can we best estimate police demand in an automated manner to inform the most effective use of police resources to reduce residential burglary in London (UK)?*

By identifying crime patterns and hotspots through predictive modeling, this project aims to enhance the effectiveness of police deployment, reduce burglary rates, and improve public safety, while considering the critical ethical implications of predictive policing.

## Key Features

-   **Interactive Geospatial Analysis**: Explore historical burglary data across London on an interactive map, with the ability to drill down from the Ward level to the LSOA (Lower Layer Super Output Area) level.
-   **Predictive Hotspot Mapping**: Visualize future burglary predictions at the Ward level, colored by risk intensity to quickly identify areas requiring attention.
-   **Data-Driven Officer Allocation**: Receive an automated recommendation for the number of police officers to allocate per shift for each ward, based on a scalable model of predicted crime.
-   **Optimized Patrol Routing**: For any selected ward, the dashboard generates an efficient, cyclical patrol route that connects predicted high-risk LSOAs using real London road networks via the OpenStreetMap API.
-   **Temporal Trend Analysis**: Analyze historical and predicted burglary trends over time with interactive monthly and yearly charts.
-   **Dynamic Filtering**: Filter the entire dashboard by year and specific wards for focused analysis.

## Technology Stack

-   **Backend & Web Framework**: [Streamlit](https://streamlit.io/)
-   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
-   **Geospatial Analysis**: [GeoPandas](https://geopandas.org/)
-   **Interactive Mapping**: [Folium](https://python-visualization.github.io/folium/)
-   **Road Network & Routing**: [OSMnx](https://osmnx.readthedocs.io/)
-   **Data Visualization**: [Altair](https://altair-viz.github.io/)

## Project Structure

```
.
├── London-wards-2018_ESRI/      # Directory for the Ward shapefiles
│   ├── London_Ward_CityMerged.shp
│   └── ... (other shapefile components)
├── data/                        # Directory for all CSV data files
│   ├── final_data.csv
│   ├── lsoa_predictions_modified.csv
│   └── future_crime_hotspot_predictions.csv
├── LSOA_2011_Boundaries.geojson # LSOA boundary data
├── website.py                   # The main Streamlit application script
└── README.md                    # This file
```

## Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   Python 3.9 or higher.
-   `pip` for package installation.

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 3. Set Up a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all required Python libraries using the provided `requirements.txt` file.

```bash
pip install streamlit pandas geopandas altair folium streamlit-folium numpy osmnx
```

### 5. Download and Place Data Files

1.  **Ward Shapefiles**: Download the "London Shape Files" from [Kaggle](https://www.kaggle.com/datasets/markjemitola/london-shape-files). Unzip the contents into a folder named `London-wards-2018_ESRI` in the project's root directory.
2.  **LSOA Boundaries**: Download the "LSOA (Dec 2011) Boundaries EW BGC" GeoJSON file from [data.gov.uk](https://www.data.gov.uk/dataset/b574a453-69bc-4fc2-88be-423486ba196d/lower-layer-super-output-areas-december-2011-boundaries-ew-bsc-v41). Rename it to `LSOA_2011_Boundaries.geojson` and place it in the project's root directory.
3.  **Project Data**: Place `final_data.csv`, `lsoa_predictions_modified.csv`, and `future_crime_hotspot_predictions.csv` into a `data` folder in the project's root directory.

After setup, your directory should match the **Project Structure** outlined above.

## Running the Application

Once all dependencies are installed and data files are in place, run the following command from your terminal in the project's root directory:

```bash
streamlit run website.py
```

Your web browser should automatically open to the dashboard.

## Methodology Overview

The project's methodology is centered on providing actionable intelligence from complex data:

1.  **Data Ingestion**: Historical crime data from `police.uk` and socioeconomic data are combined. Geographic data for London's Wards and LSOAs provide the spatial foundation.
2.  **Prediction Model**: An external model (the results of which are loaded from CSVs) predicts future burglary counts and identifies monthly hotspots at the granular LSOA level.
3.  **Aggregation**: For a strategic overview, LSOA-level predictions are aggregated up to the Ward level, which aligns with operational police districts.
4.  **Resource Allocation**: A scalable algorithm translates the total predicted crime count for a ward into a recommended number of officers per shift, providing a clear, data-driven starting point for resource planning.
5.  **Patrol Route Optimization**: For high-risk wards, the system identifies the centroids of predicted hotspots. It then uses the OSMnx library to fetch real-world road network data and calculates an efficient, cyclical patrol route connecting these hotspots using a Nearest Neighbor algorithm, ensuring a logical and practical path for officers.

## Authors

-   Pieter Pronk
-   Lia Banuta
-   Kristina Spasova
-   Gergo Racsko
-   Wouter van Abswoude
-   Mateusz Lotko

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
