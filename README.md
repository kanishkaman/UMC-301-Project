# Wildlife Conservation Monitoring System

A computer vision-based system to support wildlife conservation by classifying animal species, detecting individual animals, and analyzing behaviors. This tool helps researchers monitor animals in natural habitats, gather insights, and make informed conservation decisions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup](#setup)

## Introduction

This monitoring system aids conservationists by detecting animal species and displaying the latest data related to population and habitat.

## Features

- **Species Classification**: Identifies species using deep learning.
- **Individual Animal Re-Identification**: Detects and records animal behaviors.

## Tech Stack

- **Languages**: Python
- **ML Frameworks**: PyTorch / TensorFlow
- **LLM API**: Groq API
- **Models**: YOLOv11
- **Deployment**: Streamlit-Cloud
- **Visualization**: Matplotlib, folium

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/UMC-301-Project.git
   cd UMC-301-Project
   ```
2. **Setup the Groq API Key to use the LLM**:
   Use  `.env` file to setup your API key and import it in the `lang.py`.
   
3. **Docker**:
   ``` bash
   docker build -t my-streamlit-app .
   ```
   Run the docker image
   ``` bash
   docker run -p 8501:8501 my-streamlit-app
   ```
   You can view the website live at `localhost:8501`
