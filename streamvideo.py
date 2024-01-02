
from main import getRealTimeUpdate
import streamlit as st

# Streamlit UI
st.title("Parking Spot Tracking App")

# Video URL and Mask URL input fields
video_url = st.text_input("Enter Video URL:")
mask_url = st.text_input("Enter Mask URL:")

# Button to start tracking
start_tracking = st.button("Start Tracking")


if start_tracking:
    getRealTimeUpdate(video_url, mask_url)



