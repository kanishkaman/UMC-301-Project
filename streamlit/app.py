import streamlit as st
from PIL import Image
import os
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from species_detection.langchain.lang import chain
from species_detection.src.detect import run_pipeline
from animal_reidentification.src.detect import run_pipeline_reidentification

# Configure logging
logging.basicConfig(level=logging.INFO)
# Streamlit UI

from pages.population import render_species_distribution


def get_species_info(label):
    # Modify the prompt to get structured output
    structured_prompt = f"""
    Provide detailed information about {label} in the following structured format:
    
    HABITAT:
    [Describe natural habitat, geographical distribution, and preferred environment]

    PHYSICAL_CHARACTERISTICS:
    [Describe appearance, size, distinguishing features]

    BEHAVIOR:
    [Describe typical behavior, social structure, hunting/feeding habits]

    CONSERVATION:
    [Describe conservation status, threats, and protection efforts]

    INTERESTING_FACTS:
    [List 3-4 fascinating facts about the species]
    """
    
    # Get response from LLM
    response = chain.invoke({"question": structured_prompt})
    return response


def parse_species_info(info_text):
    """
    Parse the LLM output that uses bullet points and numbered facts
    """
    sections = {
        'habitat': {
            'description': '',
            'bullet_points': []
        },
        'physical_characteristics': {
            'description': '',
            'bullet_points': []
        },
        'behavior': {
            'description': '',
            'bullet_points': []
        },
        'conservation': {
            'description': '',
            'bullet_points': []
        },
        'interesting_facts': []
    }
    
    current_section = None
    
    # Split text into lines and process
    lines = info_text.split('\n')
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip intro and outro lines
        if line.startswith(("I'd be happy", "Would you like")):
            continue
            
        # Check for section headers
        if '**HABITAT:**' in line:
            current_section = 'habitat'
            continue
        elif '**PHYSICAL_CHARACTERISTICS:**' in line:
            current_section = 'physical_characteristics'
            continue
        elif '**BEHAVIOR:**' in line:
            current_section = 'behavior'
            continue
        elif '**CONSERVATION:**' in line:
            current_section = 'conservation'
            continue
        elif '**INTERESTING_FACTS:**' in line:
            current_section = 'interesting_facts'
            continue
            
        # Process content based on current section
        if current_section:
            if current_section == 'interesting_facts':
                # Handle numbered facts
                if line.startswith(('1.', '2.', '3.', '4.')):
                    # Remove numbering and asterisks
                    fact = line.split('.', 1)[1].strip()
                    fact = fact.replace('**', '')
                    sections['interesting_facts'].append(fact)
            else:
                # Handle bullet points and descriptions
                if line.startswith('*'):
                    # Remove asterisk and add to bullet points
                    point = line[1:].strip()
                    sections[current_section]['bullet_points'].append(point)
                else:
                    # Add to description
                    sections[current_section]['description'] += ' ' + line
    
    # Clean up descriptions
    for section in sections:
        if section != 'interesting_facts':
            sections[section]['description'] = sections[section]['description'].strip()
    
    print(f'Sections: {sections}\n')
    return sections


def main():
    # Set page config
    st.set_page_config(
        page_title="Wildlife Analysis Platform",
        page_icon="🦁",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
        }
        .info-box {
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #ddd;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title with custom styling
    st.markdown('<p class="big-font">Wildlife Analysis Platform</p>', unsafe_allow_html=True)

    # Sidebar with better organization
    with st.sidebar:
        st.image("logo.png", width=100)  # Add your logo
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Species Classification", "Habitat Mapping", "Animal Re-identification"],
            index=0
        )
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This platform helps analyze wildlife using AI.")

    # Load models
    detection_model_path = "../model/yolov8n.pt"
    classification_model_path = "../model/wildlife.pt"
    reidentification_model_path = "../model/zebra_siamese.pth"

    if page == "Species Classification":
        st.header("🔍 Species Classification")
        
        # Create two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            uploaded_image = st.file_uploader(
                "Upload an image of wildlife",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG"
            )

            if uploaded_image:
                # Save the uploaded image
                temp_image_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                # Show the uploaded image
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

                if st.button("🔎 Analyze Image"):
                    # Process image with loading animation
                    with st.spinner("Analyzing image..."):
                        buf, label, confidence = run_pipeline(
                            detection_model_path,
                            classification_model_path,
                            temp_image_path
                        )
                    if buf:
                        # Show results
                        st.image(buf, caption="Analyzed Image", use_container_width=True)
                        
                        with col2:
                            # Display classification results in a neat box
                            st.markdown("### Analysis Results")
                            with st.container():
                                st.markdown(f"""
                                    <div class="info-box">
                                        <h4>Species Identified:</h4>
                                        <p style='font-size: 20px;'>{label}</p>
                                        <h4>Confidence:</h4>
                                        <p style='font-size: 20px;'>{confidence:.2f}%</p>
                                    </div>
                                """, unsafe_allow_html=True)

                            # Get species information with loading animation
                            with st.spinner("Fetching species information..."):
                                species_info = get_species_info(label)

                                st.write(species_info)
                                print(species_info)
                                parsed_info = parse_species_info(species_info)
                    
                                # Display in organized tabs
                                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                    "🌍 Habitat", 
                                    "👀 Physical Characteristics", 
                                    "🦁 Behavior", 
                                    "🌿 Conservation", 
                                    "⭐ Fun Facts"
                                ])
                    
                    with tab1:
                        st.markdown("### Habitat")
                        habitat = parsed_info.get('habitat', 'Information not available')
                        st.markdown(habitat['description'])
                        
                    with tab2:
                        st.markdown("### Physical Characteristics")
                        physical = parsed_info.get('physical_characteristics', 'Information not available')
                        print(physical)
                        st.markdown(physical['description'])
                        
                    with tab3:
                        st.markdown("### Behavior")
                        behaviour = parsed_info.get('behavior', 'Information not available')
                        st.markdown(behaviour['description'])
                        
                    with tab4:
                        st.markdown("### Conservation Status")
                        conservation_info = parsed_info.get('conservation', 'Information not available')
                        # Add conservation status indicator
                        if 'endangered' in conservation_info['description'].lower():
                            st.error("⚠️ Endangered Species")
                        elif 'vulnerable' in conservation_info['description'].lower():
                            st.warning("⚡ Vulnerable Species")
                        elif 'near threatened' in conservation_info['description'].lower():
                            st.warning("⚡ Near Threatened Species")
                        elif 'least concern' in conservation_info['description'].lower():
                            st.success("✅ Least Concern")
                        st.markdown(conservation_info['description'])
                        
                    with tab5:
                        st.markdown("### Interesting Facts")
                        facts = parsed_info.get('interesting_facts', 'Information not available')
                        st.markdown(facts)
    if  page == "Habitat Mapping":
        species_name = st.text_input("Enter Species Name")
        render_species_distribution(species_name)
if __name__ == "__main__":
    main()