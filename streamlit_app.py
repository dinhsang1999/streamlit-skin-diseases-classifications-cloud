# --- LIBRARY ---

# Login stage
import streamlit_authenticator as stauth #pip install streamlit-authenticator
import database as db

# Fontend
import streamlit as st
from src.visual import visual_diseases,visual_dataset
from src.utils import get_image_from_lottie,crop_image,load_result,load_model,heatmap
from streamlit_lottie import st_lottie

# Backend
from PIL import Image
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad


#emoji: 
st.set_page_config(layout="wide", page_icon="👨‍🎓", page_title="Skin-classify-web-app")


# --- USER AUTHENTICATION ---
users = db.fetch_all_users()

usernames = [user['key'] for user in users]
names = [user['name'] for user in users]
hashed_password = [user['password'] for user in users]


authenticator = stauth.Authenticate(names, usernames, hashed_password,
    "skin_webapp", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    # --- PAGE TITLE ----
    image_cover = Image.open('./image/cover.png')
    st.image(image_cover,use_column_width= True)

    st.write("""
    # Skin Diseases Detect Web App

    #### This app will detect `skin diseases`

    ***Skin cancer*** is by far the world's most common cancer. Among different skin cancer types, melanoma is particularly dangerous because of its ability to metastasize. Early detection is the key to success in skin cancer treatment. However, skin cancer diagnostic is still a challenge, even for experienced dermatologists, due to strong resemblances between benign and malignant lesions. To aid dermatologists in skin cancer diagnostic, we developed a deep learning system that can effectively and automatically classify skin lesions in the ISIC dataset. An end-to-end deep learning process, transfer learning technique, utilizing multiple pre-trained models, combining with class-weighted and focal loss function was applied for the classification process. The result was that our modified famous architectures with metadata could classify skin lesions in the ISIC dataset into one of the nine classes: (1) ***Actinic Keratosis***, (2) ***Basel Cell Carcinoma***, (3) ***Benign Keratosis***, (4) ***Dermatofibroma***, (5) ***Melanocytic nevi***, (6) ***Melanoma***, (7) ***Vascular Skin Lesion*** (8) ***Squamous Cell Carcinoma*** (9) ***Unknown*** with 93% accuracy and 97% and 99% for top 2 and top 3 accuracies, respectively. This deep learning system can potentially be integrated into computer-aided diagnosis systems that support dermatologists in skin cancer diagnostic.

    ***SAVE***: 
    ```python
    - Melanoma is malinant (dangerous)
    - Others is benign (but also careful)
    ```

    ##### Some examples:
    """)
    # --- VISUAL DATA ---
    if st.checkbox('Preview examples of 9 types diseases skin'):
        st.balloons()
        visual_diseases()
    if st.checkbox('Preview dataset'):
        visual_dataset()

    # --- SIDEBAR: User input ---
    authenticator.logout("Logout","sidebar")
    
    st.sidebar.header('User Input Features')
    selected_box = st.sidebar.selectbox('Model',('Select model','Efficient_B0_256'),help="Model 1: ... - Model 2: ...")
    if selected_box == 'Select model':
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

        st_lottie(get_image_from_lottie(url = "https://assets8.lottiefiles.com/packages/lf20_mxzt4vtn.json"), key = "selectmodel", height=400)
        with st.sidebar:
            st_lottie(get_image_from_lottie(url = 'https://assets9.lottiefiles.com/private_files/lf30_zbhl9hod.json'), key='load', height=100)
    else:
        selected_image = st.sidebar.file_uploader('Upload image from PC',type=['png', 'jpg'],help='Type of image should be PNG or JPEG')
    
    if selected_box == 'Efficient_B0_256':

        if selected_image:
            if st.sidebar.checkbox('Crop image',value=True):
                crop_image = crop_image(selected_image)
                crop_image = np.array(crop_image.convert("RGB"))
                crop_image = crop_image.astype(np.int16)
            else:
                crop_image = Image.open(selected_image)
                crop_image = np.array(crop_image.convert("RGB"))
            
            st.write('##### Results:')
            load_model(selected_box)
            if st.button('Show result'):
                results = load_result(selected_box,crop_image)
                df_disease = pd.DataFrame()
                df_disease = df_disease.reset_index(drop=True)
                df_disease['diseases'] = ['MEL','NV','BCC','BKL','AK','SCC','VASC','DF','unknown']
                for i in range(5):
                    results[i][0] = np.around(results[i][0],4)*100
                    df_disease['trainer_' + str(i)] = results[i][0]
                st.dataframe(df_disease.style.highlight_max(axis=0,color='pink',subset=['trainer_0','trainer_1','trainer_2','trainer_3','trainer_4']))
                # with st.spinner("Drawing heatmap..."):
                #     image,image_ori,image_scale = heatmap(selected_box,crop_image,Cam=GradCAM)
                #     st.write('###### Original image:')
                #     st.image(image_ori)
                #     c1,c2 = st.columns(2)
                #     with c1:
                #         st.write('###### Scaled image (256x256):')
                #         st.image(image_scale)
                #     with c2:
                #         st.write('###### Heatmap image from the model:')
                #         st.image(image)
