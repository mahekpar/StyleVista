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
import google.generativeai as gai


# Load image URL from session state

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])
id = st.session_state["key"]

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

# st.set_page_config(page_title="Python Talks Search Engine", layout="wide", initial_sidebar_state="collapsed")

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


def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distence, indices = neighbors.kneighbors([features])
    return indices

# df = pd.read_csv('Fashion\Fashion-Recommender-system\pages\styles1.csv', delimiter=';')
# df = pd.read_csv('Fashion\Fashion-Recommender-system\pages\styles1.csv', delimiter=';')
try:
    df = pd.read_csv('C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/styles1.csv', delimiter=';')
    # st.write(df.head())  # Display the first few rows of the DataFrame
except Exception as e:
    st.write("Error:", e)
    
row = df[df['id'] == id]

# st.write(row)
    # Load the CSV file with error handling


def match(id):
    image_url = 'C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/' + id + '.jpg'

    # Convert image URL to BytesIO object
    uploaded_file = image_url


    if uploaded_file is not None:
            co1, co2 = st.columns(2)
            with co1:
            # Display image
                show_images = Image.open(uploaded_file)
                size = (400, 400)
                resized_im = show_images.resize(size)
                st.image(resized_im)
            
            with co2:
                st.write(f"### {row.iloc[0,9]}")
                # st.write('---')
                st.write(f"#### {'Rs. ' + str(row.iloc[0,10])}")
                st.write('Gender: ' + str(row.iloc[0,1]))
                st.write('Category: ' + str(row.iloc[0,3]))
                st.write('Article Type: ' + str(row.iloc[0,4]))
                st.write('Season: ' + str(row.iloc[0,6]))
                st.write('Occasion: ' + str(row.iloc[0,8]))
            
            
            # Extract features of uploaded image
            features = extract_img_features(uploaded_file, model)
            #st.text(features)
            img_indicess = recommendd(features, features_list)
            st.write(f"### {'Similar Products'}")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(img_files_list[img_indicess[0][1]], use_column_width =True)
            with col2:
                st.image(img_files_list[img_indicess[0][2]], use_column_width =True)
            with col3:
                st.image(img_files_list[img_indicess[0][3]], use_column_width =True)
            with col4:
                st.image(img_files_list[img_indicess[0][4]], use_column_width =True)
            with col5:
                st.image(img_files_list[img_indicess[0][5]], use_column_width =True)
    else:
        st.header("Some error occurred")
        
# Load image features and file paths
with open("image_features_embedding.pkl", "rb") as file:
    features_list = pickle.load(file)

with open("img_files.pkl", "rb") as file:
    img_files_list = pickle.load(file)

st.title('Recommendations')


        
# Load image URL from session state
id = st.session_state["key"]
match(str(id))

gai.configure(api_key='AIzaSyAAVPWfim3yEzb3dactBVsA_vKHCKQdN0M')
model = gai.GenerativeModel('gemini-pro')

matched_image_ids = []
matching_responses = []

# Dictionary to store image descriptions
image_descriptions = {}
for index, row in df.iterrows():
    image_id = row['id']
    description = row['productDisplayName']
    image_descriptions[image_id] = description

# Check if the ID exists in the image descriptions dictionary
if id in image_descriptions:
    description_for_id = image_descriptions[id]
    # print(f"Description for ID {id}: {description_for_id}")

    # Generate text for API based on the description
    text1 = 'list 5 items that can be paired with ' + description_for_id + ' without any formatting in a single sentence seperated by "," from blue jeans, black trousers, grey trousers, white shirt, black shirt, grey shirt, white tshirt, black tshirt, grey tshirt, black shoes, blue shoes, white shoes, black top, white shorts, yellow top, red dress, heels'

    # Generate content based on the text
    response1 = model.generate_content(text1)
    matching_responses.append(response1)
    # print(response1.text)
    final_response = response1.text.split(', ')

    # new_response = 'Filter the text ' + str(response1.text) + ' and return the key words as a python list '
    # response2 = model.generate_content(new_response)
    # stringResponse=str(response2)
    # print(response2.text)
    
    image_urls = {
    'blue jeans': 'fashion_small/images/51497.jpg',
    'black trousers': 'fashion_small/images/57136.jpg',
    'grey trousers': 'fashion_small/images/15116.jpg',
    'white shirt': 'fashion_small/images/11119.jpg',
    'black shirt': 'fashion_small/images/19726.jpg',
    'grey shirt': 'fashion_small/images/11117.jpg',
    'white tshirt': 'fashion_small/images/15527.jpg',
    'black tshirt': 'fashion_small/images/15347.jpg',
    'grey tshirt': 'fashion_small/images/11778.jpg',
    'black shoes': 'fashion_small/images/28636.jpg',
    'blue shoes': 'fashion_small/images/35465.jpg',
    'white shoes': "fashion_small/images/1549.jpg",
    'black top': 'fashion_small/images/1583.jpg',
    'white shorts': 'fashion_small/images/21431.jpg',
    'yellow top':'fashion_small/images/43691.jpg',
    'red dress':'fashion_small/images/57059.jpg',
    'heels':'fashion_small/images/46584.jpg'
}

# # Print image URLs based on keywords
# for keyword in final_response:
#     if keyword.lower() in image_urls:
#         print(image_urls[keyword.lower()])
#     else:
#         print(f"No image URL found for keyword: {keyword.lower()}")

    # for choice in final_response:
    #     if choice.lower() == 'blue jeans':
    #         st.image('fashion_small/images/51497.jpg')
    #     elif choice.lower() == 'black trousers':
    #         st.image('fashion_small/images/57136.jpg')
    #     elif choice.lower() == 'grey trousers':
    #         st.image('fashion_small/images/15116.jpg')
    #     elif choice.lower() == 'white shirt':
    #         st.image('fashion_small/images/11119.jpg')
    #     elif choice.lower() == 'black shirt':
    #         st.image('fashion_small/images/19726.jpg')
    #     elif choice.lower() == 'grey shirt':
    #         st.image('fashion_small/images/11117.jpg')
    #     elif choice.lower() == 'white tshirt':
    #         st.image('fashion_small/images/15527.jpg')
    #     elif choice.lower() == 'black tshirt':
    #         st.image('fashion_small/images/15347.jpg')
    #     elif choice.lower() == 'grey tshirt':
    #         st.image('fashion_small/images/11778.jpg')
    #     elif choice.lower() == 'black shoes':
    #         st.image('fashion_small/images/28636.jpg')
    #     elif choice.lower() == 'blue shoes':
    #         st.image('fashion_small/images/35465.jpg')
            
    num_images = len(final_response)
    num_rows = (num_images + 4) // 5  # Round up to the nearest multiple of 5
    st.write(f"### {'Complete the Fit with'}")
    # st.write(final_response)
    # Display the images horizontally in 5 columns
    with st.container():
        for i in range(num_rows):
            cols = st.columns(5)
            for j in range(5):
                index = i * 5 + j
                if index < num_images:
                    choice = final_response[index]
                    if choice.lower() in image_urls:
                        cols[j].image(image_urls[choice.lower()], use_column_width=True)
                    else:
                        cols[j].write(f"No image URL found for choice: {choice.lower()}")   