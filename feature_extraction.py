import tensorflow as tf
import keras as K
from keras._tf_keras.keras.applications.resnet50 import ResNet50, preprocess_input
from keras._tf_keras.keras.layers import GlobalMaxPooling2D, MaxPooling2D
import dask, distributed

import cv2,os,pickle
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass, Field
from typing import Any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def show_image(image_path:os.PathLike):
    img = cv2.imread(image_path)
    # reshape it to 224x224x3
    img = cv2.resize(src=img,dsize=(224,224)) 
    file_name = os.path.split(image_path)[0]
    cv2.imshow(file_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@dataclass
class Feature_Model:

    weights:str='imagenet'
    input_shape:Any=(224,224,3)
    include_top:bool=False

    def __post_init__(self):
        model = ResNet50(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape)
        # as the resnet model is already trained on another dataset, set trainable to False
        model.trainable = False 
        self.model = K.Sequential([
            model, 
            # using MaxPooling2D will return 3x3x2048, using GlobalMaxPooling2D would return a vector of 2048
            GlobalMaxPooling2D() 
        ])
    
    def model_(self):
        return self.model
    

@dataclass
class Feature_Extraction:
    '''
    functions: extract_feature(s.method), extract_feature_from_folder(method), save_(s.method)

    The extract_feature_from_folder method uses dask.delayed for parallel computation.
    '''
    @staticmethod
    def extract_feature(img_path:os.path, model:ResNet50)->np.array:
        # Load the image
        img = cv2.imread(img_path) 
        # Resize the image to 224 x 224 x 3; 3 being the channel
        img = cv2.resize(img,dsize=(224,224))
        # Expand the image
        img = np.expand_dims(a=img, axis=0)
        # Preprocess the image
        img = preprocess_input(img)
        # Predict the image
        img = model.predict(img, verbose='3')
        # Flatten the output from 1,2048 to 2048
        img = img.flatten()
        # Normalize the output
        img = img / np.linalg.norm(img)
        
        return img

    def extract_feature_from_folder(self, model:K.Model,folder:os.path.dirname):
        feature_list =[]
        # get all image_paths in the provided folder
        filenames = [ os.path.join(folder, filename) for filename in tqdm(os.listdir(folder),desc='Appending files') ]
        # extract features
        feature_list = [ dask.delayed(self.extract_feature)(file_path, model) for file_path in tqdm(filenames, desc= 'Generating and appending features') if file_path.endswith('.jpg')  ]
        # return tuple of feature_list, filenames
        features = dask.compute(feature_list)
        return features, filenames
    
    @staticmethod
    def save_(model:K.Model, features_list:list, filenames:list)->None:
        try:
            # creates checkpoint if not exists
            if not os.path.exists('checkpoint'):
                os.makedirs('checkpoint', exist_ok=True)
            # save model
            model.save('checkpoint/model.keras')

            # pickle the features_list, filenames
            with open('checkpoint/featurevectors.pkl','wb') as file:
                pickle.dump(features_list, file=file, protocol=pickle.HIGHEST_PROTOCOL)

            with open('checkpoint/image_paths.pkl','wb') as file:
                pickle.dump(filenames, file=file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            raise
        
if __name__=="__main__":
    __all__ = ["Feature_Model","Feature_Extraction","show_image"]