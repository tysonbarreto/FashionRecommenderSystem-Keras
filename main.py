from feature_extraction import Feature_Extraction, Feature_Model, show_image
from cluster import Cluster
import pickle
import numpy as np
import cv2, os
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image


# load feature vecotrs and filenames pickle file in checkpoint
feature_list = pickle.load(open('checkpoint/featurevectors.pkl','rb'))
filenames = pickle.load(open('checkpoint/image_paths.pkl','rb'))

# convert to numpy arrays
feature_list = np.array(feature_list).squeeze(axis=0)
filenames = filenames

# load ResNet50 Model
fe = Feature_Extraction()
model = Feature_Model().model_()

#print(model.summary())
#print(filenames[:5])
#print(feature_list[0].shape)

neighbors= Cluster(feature_list=feature_list)
#normalized_img = fe.extract_feature(img_path=r'dataset/1636.jpg',model=model)
#indices = neighbors.ClassifyNeighbors(matrix_like=[normalized_img])

#file_paths = [filenames[file_index] for file_index in indices[0]]

def save_uploaded_file(uploaded_file:UploadedFile):
    if not os.path.exists('uploads'):
        os.makedirs('uploads', exist_ok=True)
    try:
        with open(os.path.join('uploads',uploaded_file.name), 'wb') as file:
            file.write(uploaded_file.getbuffer())
        return True
    except Exception:
        raise 

def App():

    st.title("Fashion Recommender System")

    uploaded_file = st.file_uploader("Uploade an image")
    print(uploaded_file)

    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file=uploaded_file):
            # show image
            display_image = Image.open(uploaded_file)

            # resize image
            resized_image = display_image.resize(size=(200,200))

            # get features and corresponding indices
            features = fe.extract_feature(img_path=os.path.join('uploads',uploaded_file.name),model=model)
            indices = neighbors.ClassifyNeighbors(matrix_like=[features])

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.image(filenames[indices[0][0]])
            with col2:
                st.image(filenames[indices[0][1]])
            with col3:
                st.image(filenames[indices[0][2]])
            with col4:
                st.image(filenames[indices[0][3]])
            with col5:
                st.image(filenames[indices[0][4]])
        else:
            st.header("Error occured while uploading the file, please check file/file format and retry...")

if __name__=="__main__":
    App()





