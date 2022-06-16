# --- LIBRARY ---
from json import load

import requests
import streamlit as st
from streamlit_lottie import st_lottie


# --- LOTTIE ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def get_image_from_lottie(url=None,filepath=None):
    '''
    Augs:
        - url (String): url from lottie website
        - filepath (String): path to lottie file
    Return: 
        image
    Note:
        - Choose one: url or filepath, not both. If both, the function will be return image from url
    '''
    if url != None:
        return load_lottieurl(url)
    elif filepath != None:
        return None #FIXME:
    else:
        return None

def seleted_model(model='Select model'):

    if model == 'Select model':
        st.markdown("""
        <span style = 'font-size:30px;'> 
        let's sellect
        </span>
        <span style = 'color:pink;font-size:40px;'>
        Model
        </span>
        <span style = 'font-size:30px;'> 
        !
        </span>
        """,
        unsafe_allow_html=True)

        st_lottie(get_image_from_lottie("https://assets8.lottiefiles.com/packages/lf20_mxzt4vtn.json"), key = "selectmodel", height=400)
        with st.sidebar:
            st_lottie(get_image_from_lottie('https://assets9.lottiefiles.com/private_files/lf30_zbhl9hod.json'), key='load', height=100)