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
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load image features and file paths
with open("image_features_embedding.pkl", "rb") as file:
    features_list = pickle.load(file)

with open("img_files.pkl", "rb") as file:
    img_files_list = pickle.load(file)
    
# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

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



def match(id):
    image_url = 'C:/Users/Vivek/Desktop/codes/Fashion/Fashion-Recommender-system/fashion_small/images/' + id + '.jpg'

    # Convert image URL to BytesIO object
    uploaded_file = image_url


    if uploaded_file is not None:
            # Display image
            show_images = Image.open(uploaded_file)
            size = (400, 400)
            resized_im = show_images.resize(size)
            # st.image(resized_im)
            # Extract features of uploaded image
            features = extract_img_features(uploaded_file, model)
            #st.text(features)
            img_indicess = recommendd(features, features_list)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(img_files_list[img_indicess[0][1]])
            with col2:
                st.image(img_files_list[img_indicess[0][2]])
            with col3:
                st.image(img_files_list[img_indicess[0][3]])
            with col4:
                st.image(img_files_list[img_indicess[0][4]])
            with col5:
                st.image(img_files_list[img_indicess[0][5]])
    else:
        st.header("Some error occurred")


st.title("Hello Mahek!")
st.subheader("Clothing Search")
c1,c2 = st.columns([5,1])
with c1:
    text_search = st.text_input("Search your choice", value="")

with c2:
    option = st.selectbox("",
    ("High to Low", "Low to High"),
    index=None,
    placeholder="Sort Price by",
    )


# df = pd.DataFrame({'Username' : ['Omkar', 'Vivek', 'Mahek', 'Darsh'], 
#                    'Purchasehistory': [['1163', '1525', '1531', '1543', '1569', '1880', '2009'], 
#                                        ['2135', '2007', '1641', '1551', '2041', '2243', '2009'],
#                                        ['2296', '2562', '2576', '15098', '15106', '42309', '45777'],
#                                        ['42260', '42267', '42275', '3016', '3052', '3777', '7049.jpg']]})

# user_name=st.text_input("Enter Username",value="")
# df.index=['Omkar','Vivek','Mahek','Darsh']
# # st.write(df)
# history=df.loc[user_name,'Purchasehistory']

st.write(f"#### {'Based on your previous purchases we recommend:'}")
match('2296')


if text_search:
    nltk.download('stopwords')
    text_search = re.sub('[^a-zA-Z]',' ', text_search)
    text_search = text_search.lower()
    text_search = text_search.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    text_search = [ps.stem(word) for word in text_search if not word in set(all_stopwords)]
    text_search = ' '.join(text_search)
    # st.write(text_search)

df = pd.read_csv('styles1.csv', delimiter=';')


def word_search(text_search, df):
    text_search = str(text_search.lower())
    text_search = text_search.split()
    matched_ids = []
    for index, row in df.iterrows():
        product_name = str(row['productDisplayName'])
        product_name = product_name.lower()
        words_list = product_name.split()
        count = 0
        for term in text_search:
            if term in words_list:
                count += 1
                if count >= 3:  # At least two words match
                    matched_ids.append(row['id'])
                    break  # Exit the loop once two words match
    return matched_ids

if text_search:
    matched_ids = word_search(text_search, df.copy())  

    if option =='High to Low':
        matched_ids.sort(key=lambda x: df.loc[df['id'] == x, 'price'].iloc[0], reverse= True)
    else:
        matched_ids.sort(key=lambda x: df.loc[df['id'] == x, 'price'].iloc[0], reverse=False)


    N_cards_per_row = 3
    c=16
    for i, id in enumerate(matched_ids):
        if i % N_cards_per_row == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        row_index = df[df['id'] == id].index[0]  
        c=c-1
        if c==0:
            break
        with cols[i%N_cards_per_row]:
            # st.markdown(f"**{df.loc[row_index, 'productDisplayName'].strip()}**")
            # image_url = 'images/'+ str(id) + '.jpg'
            # st.markdown(image_url)
            # st.image(image_url)
            title = df.loc[row_index, 'productDisplayName'].strip()
            
            image_url = 'fashion_small/images/' + str(id) + '.jpg'
            st.image(image_url,  use_column_width=True)
            st.markdown(f"##### {title}")
            st.write(f"##### {'Rs. ' + str(df.loc[row_index,'price'])}")
            # if st.button(f"Go to {i+1}"):
            #     st.write(id)
            #     st.session_state["key"] = id
            #     st.page_link('pages/page.py')
            # if st.button(f"Go to {i+1}"):
            if st.button(f"Go to {i+1}"):
                st.session_state["key"] = id
                st.experimental_rerun()
