import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the trained model
with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('breastcancer.csv')
# Streamlit page config
st.set_page_config(page_title='🩺 Breast Cancer Predictor', layout='centered')
st.title('🧬 Breast Cancer Prediction App')
st.markdown('#### Provide tumor feature details below:')

# Sidebar input sliders (default values set to benign-like values)
radius_mean = st.sidebar.slider('Radius Mean', 5.0, 30.0, 12.0)
texture_mean = st.sidebar.slider('Texture Mean', 9.0, 40.0, 14.0)
perimeter_mean = st.sidebar.slider('Perimeter Mean', 40.0, 190.0, 78.0)
area_mean = st.sidebar.slider('Area Mean', 150.0, 2500.0, 450.0)
smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.05, 0.2, 0.08)
compactness_mean = st.sidebar.slider('Compactness Mean', 0.02, 1.0, 0.07)
concavity_mean = st.sidebar.slider('Concavity Mean', 0.0, 1.0, 0.04)
concave_points_mean = st.sidebar.slider('Concave Points Mean', 0.0, 0.3, 0.03)
symmetry_mean = st.sidebar.slider('Symmetry Mean', 0.1, 0.4, 0.18)
fractal_dimension_mean = st.sidebar.slider('Fractal Dimension Mean', 0.04, 0.1, 0.06)

# Construct DataFrame with column names matching training
input_data = pd.DataFrame([{
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'perimeter_mean': perimeter_mean,
    'area_mean': area_mean,
    'smoothness_mean': smoothness_mean,
    'compactness_mean': compactness_mean,
    'concavity_mean': concavity_mean,
    'concave_points_mean': concave_points_mean,  # column with space
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean
}])

# Optional: Display input data for debugging
st.subheader("🔎 Input Data Preview")
st.dataframe(input_data)

# Prediction logic
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f'⚠️ Diagnosis: **Malignant** ({probability * 100:.2f}%)')
        else:
            st.success(f'✅ Diagnosis: **Benign** ({probability * 100:.2f}%)')

    except Exception as e:
        st.error(f"🚨 Prediction failed:\n\n{e}")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
local_css('styles.css')

import plotly.graph_objects as go
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean'
]

# User inputs as a list
values = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
          compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
          fractal_dimension_mean]

# Plot radar chart
st.subheader("📈 Input Feature Radar Chart")

fig = go.Figure(data=go.Scatterpolar(
    r=values,
    theta=features,
    fill='toself',
    line_color='purple'
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=False
)

st.plotly_chart(fig)