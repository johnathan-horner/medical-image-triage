#!/usr/bin/env python3
"""Simple test app to diagnose Streamlit Cloud issues"""

import streamlit as st
import os

st.title("🔧 Deployment Test")

st.write("## Environment Check")
st.write(f"Python version: {st.__version__}")
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in directory:")

for file in os.listdir("."):
    st.write(f"- {file}")

st.write("## Import Test")
try:
    import requests
    import pandas as pd
    import plotly
    import numpy as np
    from PIL import Image
    st.success("✅ All imports successful!")
except Exception as e:
    st.error(f"❌ Import error: {e}")

st.write("## File Check")
if os.path.exists("docs/Medical_Image_Triage_AWS_Architecture.png"):
    st.success("✅ Architecture diagram found")
    st.image("docs/Medical_Image_Triage_AWS_Architecture.png", caption="Architecture", width=300)
else:
    st.error("❌ Architecture diagram missing")

if os.path.exists("samples/normal_chest_xray.png"):
    st.success("✅ Sample images found")
else:
    st.error("❌ Sample images missing")

st.write("## App Test")
if st.button("Test Button"):
    st.balloons()
    st.success("Button works!")