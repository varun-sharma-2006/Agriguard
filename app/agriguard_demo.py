import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import folium
from streamlit_folium import st_folium
from auth import register_user, authenticate_user

# Setup Path for imports
sys.path.append('.')
from src.models.vision_inference import ImagePredictor

@st.cache_resource
def load_vision_model():
    return ImagePredictor(model_path='models/vision_model.pth')

def dashboard_app():
    st.title("🌾 AgriGuard: Computer Vision Disease Detection")
    st.markdown("**Upload Sentinel-2 imagery or direct crop photos for instant ResNet classification.**")
    st.markdown("---")
    
    predictor = load_vision_model()
    
    # Sidebar
    st.sidebar.header("🗺️ Context Data")
    lat = st.sidebar.number_input("Latitude (Optional)", value=15.3173, format="%.4f")
    lon = st.sidebar.number_input("Longitude (Optional)", value=75.7139, format="%.4f")
    
    # Image Upload Section
    st.header("📸 Image Analysis")
    st.markdown("Upload a close-up photo of the crop leaf, or a raw multispectral crop extract.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Sensory Data", use_column_width=True)
            
        with col2:
            with st.spinner("Analyzing spectral data via PyTorch ResNet..."):
                bytes_data = uploaded_file.getvalue()
                result = predictor.predict_image(bytes_data)
                
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            probs = result['probabilities']
            
            if predicted_class == 'model_not_trained':
                st.error("⚠️ The model has not been trained yet. Please run src/data_pipeline/download_dataset.py and src/models/vision_cnn.py first to build vision_model.pth!")
                st.stop()
                
            st.subheader("🎯 Diagnosis Results")
            status_emoji = "🔴" if 'disease' in predicted_class.lower() else "🟢" if 'health' in predicted_class.lower() else "🟡"
            
            met1, met2 = st.columns(2)
            met1.metric("Status", f"{status_emoji} {predicted_class.title()}")
            met2.metric("Confidence", f"{confidence:.1%}")
            
            # Action Suggestions
            if 'disease' in predicted_class.lower() or predicted_class.lower() in ['diseased', 'blight', 'rust']:
                st.error("🚨 **Immediate action required!** Pathogen detected. Formulate fungicide deployment.")
            elif 'stress' in predicted_class.lower():
                st.warning("⚠️ **Stress Detected.** Monitor water lines and nutrient deposits.")
            else:
                st.success("✅ **Crop appears healthy.** Continue standard regimens.")
                
            # Class Probabilities Bar Chart
            st.subheader("📊 Network Confidence Distribution")
            
            fig_probs = go.Figure(go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=['green' if 'health' in k.lower() else 'red' if 'disease' in k.lower() else 'orange' for k in probs.keys()],
                text=[f'{v:.1%}' for v in probs.values()],
                textposition='auto'
            ))
            fig_probs.update_layout(yaxis_title="Probability", yaxis_range=[0, 1], height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig_probs, use_container_width=True)
            
        # Map integration below
        st.markdown("---")
        st.subheader("🌍 Registered Field Location")
        m = folium.Map(location=[lat, lon], zoom_start=14)
        folium.Marker(
            [lat, lon], 
            popup=f"Scan Result: {predicted_class.title()}",
            icon=folium.Icon(color="red" if 'disease' in predicted_class.lower() else "green")
        ).add_to(m)
        st_folium(m, width=900, height=350)
        
    else:
        st.info("👆 Please upload an image file to begin inference.")

def main():
    st.set_page_config(
        page_title="AgriGuard Vision Deep Learning",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = None

    if st.session_state['authenticated']:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.rerun()
        dashboard_app()
    else:
        st.title("Welcome to AgriGuard Vision 📸")
        st.markdown("Please log in or sign up to access the computer vision dashboard.")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login")
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    success, msg = authenticate_user(login_username, login_password)
                    if success:
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = login_username
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                    
        with tab2:
            st.subheader("Sign Up")
            with st.form("signup_form"):
                signup_username = st.text_input("Choose Username")
                signup_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Sign Up")
                
                if submitted:
                    if signup_password != confirm_password:
                        st.error("Passwords do not match!")
                    elif len(signup_password) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        success, msg = register_user(signup_username, signup_password)
                        if success:
                            st.success(msg + " Please switch to the Login tab.")
                        else:
                            st.error(msg)

if __name__ == "__main__":
    main()