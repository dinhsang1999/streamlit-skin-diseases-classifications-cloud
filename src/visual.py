import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def visual_diseases():
    st.write('`Melanoma`')
    image_mel = Image.open('./image/MEL.png')
    st.image(image_mel)
    st.write('Actinic Keratosis')
    image_ak = Image.open('./image/AK.png')
    st.image(image_ak)
    st.write('Basel Cell Carcinoma')
    image_bcc = Image.open('./image/BCC.png')
    st.image(image_bcc)
    st.write('Benign Keratosis')
    image_bkl = Image.open('./image/BKL.png')
    st.image(image_bkl)
    st.write('Dermatofibroma')
    image_df = Image.open('./image/DF.png')
    st.image(image_df)
    st.write('Melanocytic nevi')
    image_nv = Image.open('./image/NV.png')
    st.image(image_nv)
    st.write('Vascular Skin Lesion')
    image_vasc = Image.open('./image/VASC.png')
    st.image(image_vasc)
    st.write('Squamous Cell Carcinoma')
    image_scc = Image.open('./image/SCC.png')
    st.image(image_scc)
    st.write('Unknown')
    image_ukn = Image.open('./image/unknown.png')
    st.image(image_ukn)

def visual_dataset():
    df = pd.read_csv('./csvfile/train.csv')
    df_fulltrain = pd.read_csv('./csvfile/full_train.csv')
    df_test = pd.read_csv('./csvfile/test.csv')
    df = df.drop(columns=['Unnamed: 0'],axis=1)
    st.dataframe(df.head(10))
    l_df = len(df)
    l_df_test = len(df_test)
    ll = f"""
    Length of training dataset: **<span style = 'font-size:20px;text-decoration:underline;'>{l_df}</span>** images \n
    Length of validate dataset: **<span style = 'font-size:20px;text-decoration:underline;'>{round(l_df/5)}</span>** images \n
    Length of testing dataset: **<span style = 'font-size:20px;text-decoration:underline;'>{l_df_test}</span>** images \n

    Length of melanoma images: **<span style = 'font-size:20px;text-decoration:underline;'>{df['target'].value_counts()[1]}</span>** images \n
    """
    st.markdown(ll,
        unsafe_allow_html=True)
    st.write('Before:')
    st.write(df['diagnosis'].value_counts())
    st.write('After:')
    st.write(df_fulltrain['diagnosis'].value_counts())
    fig1 = plt.figure(figsize=(10, 4))
    sns.countplot(data=df,x = "sex",palette="Set2")
    st.pyplot(fig1)
    fig2 = plt.figure(figsize=(10, 2))
    sns.countplot(data=df,x = "age_approx")
    st.pyplot(fig2)
    fig3 = plt.figure(figsize=(15, 8))
    sns.countplot(data=df,x = "anatom_site_general_challenge")
    st.pyplot(fig3)
