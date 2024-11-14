# Wildlife Species Detection and Data Visualization

This part of the project aims to use the identified wildlife species from our model to retrieve additional information, and visualize global sightings and few other plots. It utilizes the custom-trained YOLOv11 for species detection, Wikipedia API for species information, Generative AI for enhanced details, and GBIF/iNaturalist API for species occurrence data. The project also includes interactive mapping capabilities for visualizing species distribution on a world map.

> **Note**: These final files are intended for implementation in Streamlit, with some modifications in the information retrieval process. In the original Streamlit code, LangChain with Ollama was used instead of the Gemini API, due to module/package compatibility issues.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Overview](#usage-overview)
- [Features and Technologies Used](#features-and-technologies-used)
- [Contributing](#contributing)

## Installation

1. **Clone the repository**.

2. **Install dependencies**.

3. **Data and Model Setup**:
    - Download the YOLOv11 weights file (`wildlife.pt`) and save it in the root directory.
    - Ensure the Natural Earth shapefiles for countries (`ne_110m_admin_0_countries.shp`) are stored in the specified directory (`../data_retrieval/ne_110m_admin_0_countries/`).

## Project Structure

- **`lang_vis.ipynb`** - Jupyter Notebook containing the main code for species detection, information retrieval, and data visualization.
- **`wildlife.pt`** - Custom-trained YOLOv11 weights for species detection.
- **`results/`** - Directory to store outputs, including:
  - **`enhanced_info.txt`** - Text file with enhanced species information.
  - **`plots`** -  Saved plots showing various visualisations generated while running the code.
  - **`sightings_map.html`** - HTML file of the interactive map generated for species distribution.
- **`gbif_data.csv`** - Output file storing GBIF species occurrence data.
- **`trial_image.png`** - Sample image (of a zebra) for testing the code in `lang_vis.ipynb`.


## Usage Overview

1. **Prepare the Environment**: Ensure you have installed all the necessary libraries (create a venv if possible).

2. **Run the Code**:
   - Open the `lang_vis.ipynb` file in Jupyter Notebook.
   - Load the sample image `trial_image.png` or use your own image for species prediction. (set this in the `image_path`)
   - Execute the cells in sequence to:
     - Predict the species using YOLOv11 weights.
     - Retrieve detailed species information using the Wikipedia API and enhance it with Gemini API.
     - Fetch species/subspecies occurrence data from the GBIF/iNaturalist API and plot observations.
     - Generate interactive maps and save them in HTML format.

3. **Configure API Keys**:
   Set your API key in the `enhance_species_info` function.
   
   (Update paths for `wildlife.pt`, `gbif_data.csv`, and shapefile (`ne_110m_admin_0_countries.shp`), if needed.)

5. **Store Results**:
   - Enhanced species information will be saved in the `results/enhanced_info.txt` file.
   - Plots of species observations by country will be saved in the `results/` folder.
   - Interactive HTML maps showing species distributions will also be saved there.

4. **Testing**:
   - Use the sample image `trial_image.png` as a quick test to verify that the setup is working correctly and see sample outputs.


## Features and Technologies Used

1. **Species Detection**:
   - Utilizes YOLOv11 to identify the species in the provided image.
   
2. **Information Retrieval**:
   - Fetches species information from Wikipedia and enhances it using Gemini AI.
   - In the Streamlit version, LangChain with Ollama is used instead.
   
3. **Species Occurrence Data**:
   - Retrieves species occurrence data from GBIF/iNaturalist API and saves it to a CSV file.
   - Filters the data to later create various plots using Matplotlib and Seaborn.
   
4. **Maps and Plots**:
   - Plots sightings by country and maps global distribution using geopandas (and folium).

## Contributing

This part of the project has been entirely contributed by me, Kanishk Aman. Any more contributions are welcome! Please open an issue or submit a pull request.

