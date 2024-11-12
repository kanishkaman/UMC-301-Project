
# Use the official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenGL and others
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY ./streamlit/assets/logo.png /app/streamlit/assets/logo.png
COPY . /app


# Expose the port Streamlit will run on
EXPOSE 8501

# Set the environment variable to prevent Python from writing .pyc files to disc
ENV PYTHONUNBUFFERED 1

# Command to run your Streamlit app
CMD ["streamlit", "run", "streamlit/app.py"]
