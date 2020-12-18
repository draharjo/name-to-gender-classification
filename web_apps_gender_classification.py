import streamlit as st
from jcopml.utils import load_model

import pandas as pd
import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize

st.title('Name to gender classification')
st.text("This apps is predict whether a name is male or female")

#Load FastText model & ML classifier
w2v = FastText.load("model/fasttext/name_gender.fasttext").wv
model = load_model('model/name_to_gender.pkl')

#Create feature extraction
def norm_sent_vector(sentence, w2v_model):
    vecs = [w2v_model[word.lower()] for word in word_tokenize (sentence)]    
    norm_vecs = [vec/ np.linalg.norm(vec) for vec in vecs if np.linalg.norm(vec > 0)]
    sent_vec = np.mean (vecs, axis = 0) #dari arah baris
    return sent_vec

name_str = st.text_input("Name to predict")
name = [name_str]

if name is not None:
    try:
        vecs_name = [norm_sent_vector (sentence.lower(), w2v) for sentence in name]
        vecs_name = np.array(vecs_name)
        prediction = model.predict(vecs_name)[0]
        if prediction == "L":
            prediction = "male"
        else:
            prediction = "female"
        st.text(f"\" %s \" is predicted to be a %s name" %(name[0],prediction))

    except:
        pass