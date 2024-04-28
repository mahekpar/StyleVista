import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from io import BytesIO


st.set_page_config(page_title="Python Talks Search Engine", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

c1,c2,c3 = st.columns(3)
with c1:
    co1,co2,co3 = st.columns([1,2,1])
    with co2:
        st.page_link("Search.py", label = "Search", icon = "🔎", use_container_width = True)

with c2:
    co1,co2,co3 = st.columns([1,2,1])
    with co2:
        st.page_link("pages/History.py", label = "Order History", icon = "📃", use_container_width = True)

with c3:
    co1,co2,co3 = st.columns([1,2,1])
    with co2:
        st.page_link("pages/Recommendations.py", label = "Recommendations", icon = "✅", use_container_width=True)
        

st.title("Your Past Purchases:")


df = pd.read_csv('C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/styles1.csv', delimiter=';')


images= ['2296', '2562', '2576', '15098', '15106']
# c1,c2,c3,c4,c5=st.columns(5)
# with c1:
#     st.image("C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/"+images[0]+'.jpg', use_column_width=True)
#     row = df[df['id'] == id]
#     st.write(f"#### {'Rs. ' + str(row.iloc[0,10])}")

#     st.write()
# with c2:
#     st.image("C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/"+images[1]+'.jpg', use_column_width=True)
#     row = df[df['id'] == id]
#     st.write(f"#### {'Rs. ' + str(row.iloc[0,10])}")
# with c3:
#     st.image("C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/"+images[2]+'.jpg', use_column_width=True)
#     row = df[df['id'] == id]
#     st.write(f"#### {'Rs. ' + str(row.iloc[0,10])}")
# with c4:
#     st.image("C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/"+images[3]+'.jpg', use_column_width=True)
#     row = df[df['id'] == id]
#     st.write(f"#### {'Rs. ' + str(row.iloc[0,10])}")
# with c5:
#     st.image("C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/"+images[4]+'.jpg', use_column_width=True)
#     row = df[df['id'] == id]
#     st.write(f"#### {'Rs. ' + str(row.iloc[0,10])}")
    
# images = ['2296', '2562', '2576', '15098', '15106']
images = ['2296', '2562', '2576', '15098', '15106']

# Define the number of columns to display the images horizontally
num_columns = len(images)

# Create columns to display the images horizontally
columns = st.columns(num_columns)
purchased_on = ['April 14, 2024','April 16, 2024','April 18, 2024','April 20, 2024','April 22, 2024']
# Loop through the images and display each image in its corresponding column
for i, image_id in enumerate(images):
    row = df[df['id'] == int(image_id)]  # Assuming 'id' column contains integers
    if not row.empty:
        with columns[i]:
            st.image("C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/" + image_id + '.jpg', use_column_width=True)
            st.write(f"##### {row.iloc[0,9]}")
            st.write(f"###### {'Rs. ' + str(row.iloc[0, 10])}")
            st.write(f"###### Purchased on: {purchased_on[i]}")