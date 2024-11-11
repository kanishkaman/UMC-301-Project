import streamlit as st
from PIL import Image
import os
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import sys
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import altair as alt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)



def render_species_distribution():
    st.header("🗺️ Species Distribution Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Global Distribution", "Sightings Timeline", "Statistics"])

    # Fetch data with loading indicator
    with st.spinner("Fetching species data..."):
        species_name = 'panthera pardus'  # You can make this dynamic based on detected species
        iNaturalist_data = get_inaturalist_data(species_name)
        gbif_data = get_gbif_data(species_name)

    # Tab 1: Distribution Map
    with tab1:
        st.subheader("Global Sightings Distribution")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create and display map
            if iNaturalist_data or gbif_data:
                map_obj = plot_sightings_map(iNaturalist_data, gbif_data)
                st_folium(map_obj, width=700, height=500)
            else:
                st.warning("No distribution data available for this species")
        
        with col2:
            st.markdown("### Data Sources")
            st.markdown("""
                - 🟢 iNaturalist Sightings
                - 🔵 GBIF Records
            """)
            
            # Display counts
            st.metric("iNaturalist Sightings", len(iNaturalist_data))
            st.metric("GBIF Records", len(gbif_data))

    # Tab 2: Timeline Analysis
    with tab2:
        st.subheader("Sightings Timeline")
        
        # Process dates for timeline
        dates_data = process_dates(iNaturalist_data, gbif_data)
        if dates_data:
            fig = plot_timeline(dates_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No timeline data available")

    # Tab 3: Statistics
    with tab3:
        st.subheader("Sighting Statistics")
        show_statistics(iNaturalist_data, gbif_data)

def get_inaturalist_data(species_name):
    """Fetch data from iNaturalist API"""
    try:
        url = f"https://api.inaturalist.org/v1/observations?taxon_name={species_name}&per_page=50"
        response = requests.get(url)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        st.error(f"Error fetching iNaturalist data: {str(e)}")
        return []

def get_gbif_data(species_name):
    """Fetch data from GBIF API"""
    try:
        gbif_search_url = f"https://api.gbif.org/v1/species?name={species_name}"
        search_response = requests.get(gbif_search_url)
        search_data = search_response.json()
        
        if search_data.get("results"):
            species_key = search_data["results"][0]["key"]
            occurrences_url = f"https://api.gbif.org/v1/occurrence/search?taxonKey={species_key}&limit=50"
            occurrences_response = requests.get(occurrences_url)
            return occurrences_response.json().get("results", [])
        return []
    except Exception as e:
        st.error(f"Error fetching GBIF data: {str(e)}")
        return []

def plot_sightings_map(iNaturalist_data, gbif_data):
    """Create a Folium map with sightings data"""
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Plot iNaturalist data
    for obs in iNaturalist_data:
        if obs.get("geojson") and obs["geojson"].get("coordinates"):
            coords = obs["geojson"]["coordinates"]
            folium.Marker(
                location=[coords[1], coords[0]],
                popup=obs.get("species_guess", "Unknown Species"),
                icon=folium.Icon(color="green")
            ).add_to(m)
    
    # Plot GBIF data
    for obs in gbif_data:
        if "decimalLatitude" in obs and "decimalLongitude" in obs:
            folium.CircleMarker(
                location=[obs["decimalLatitude"], obs["decimalLongitude"]],
                radius=3,
                color="blue",
                fill=True,
                fill_opacity=0.6
            ).add_to(m)
    
    return m

def process_dates(iNaturalist_data, gbif_data):
    """Process dates from both data sources for timeline visualization"""
    dates = []
    
    # Process iNaturalist dates
    for obs in iNaturalist_data:
        if obs.get("observed_on"):
            dates.append({
                'date': obs["observed_on"],
                'source': 'iNaturalist'
            })
    
    # Process GBIF dates
    for obs in gbif_data:
        if obs.get("eventDate"):
            dates.append({
                'date': obs["eventDate"].split('T')[0],
                'source': 'GBIF'
            })
    
    return pd.DataFrame(dates)

def plot_timeline(df):
    """Create timeline visualization using Plotly"""
    df['date'] = pd.to_datetime(df['date'])
    df_grouped = df.groupby(['date', 'source']).size().reset_index(name='count')
    
    fig = px.line(df_grouped, x='date', y='count', color='source',
                  title='Sightings Over Time',
                  labels={'date': 'Date', 'count': 'Number of Sightings'},
                  color_discrete_map={'iNaturalist': 'green', 'GBIF': 'blue'})
    
    return fig

def show_statistics(iNaturalist_data, gbif_data):
    """Display statistical information about sightings"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### iNaturalist Statistics")
        if iNaturalist_data:
            countries = pd.DataFrame([
                obs.get('place_guess', 'Unknown')
                for obs in iNaturalist_data
            ]).value_counts().head()
            
            st.bar_chart(countries)
    
    with col2:
        st.markdown("### GBIF Statistics")
        if gbif_data:
            countries = pd.DataFrame([
                obs.get('country', 'Unknown')
                for obs in gbif_data
            ]).value_counts().head()
            
            st.bar_chart(countries)

