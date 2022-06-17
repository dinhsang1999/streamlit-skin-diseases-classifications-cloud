# --- LIBRARY ---
from json import load

import requests
import streamlit as st
import numpy as np
import timm
import torch
import os
import urllib
import cv2
import random
from PIL import Image
import albumentations as A
from src.model import MelanomaNet,BaseNetwork,MetaMelanoma
from streamlit_cropper import st_cropper
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


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
    return

def crop_image(image):
    '''
    '''
    img = Image.open(image)
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
    crop_image = st_cropper(img,aspect_ratio=(1,1),box_color='#0000FF',realtime_update=realtime_update)
    return crop_image

def draw_heatmap(model_name,image,Cam=GradCAM):
    '''
    '''
    model = timm.create_model(model_name,image,Cam=GradCAM)

@st.cache(allow_output_mutation=True,ttl=3600*24,max_entries=1,show_spinner=False)
def load_model(model_name):
    if model_name == 'Efficient_B0_256':
        model_name = 'efficientnet_b0'
        os.makedirs('model',exist_ok = True)
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            url_init = 'https://github.com/dinhsang1999/streamlit-skin-diseases-classifications-cloud/releases/download/efficientnet_b0/'
            for i in range(5):
                url = url_init + 'efficientnet_b0_fold' + str(i) + '.pth'
                path_out = os.path.join('model','efficientnet_b0_fold' + str(i) + '.pth')
                urllib.request.urlretrieve(url, path_out)

def load_result(model_name,image,meta_features=None):
    '''
    '''
    accuracy_5 = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == 'Efficient_B0_256':
        with st.spinner("Calculating results..."):
            for i in range(5):
                model = BaseNetwork('efficientnet_b0')
                model.to(device)
                path_model = os.path.join('model', 'efficientnet_b0_fold' + str(i) + '.pth')
                model_loader = torch.load(path_model,torch.device('cpu'))
                #delete modul into model to train 1 gpu
                model_loader = {key.replace("module.", ""): value for key, value in model_loader.items()}
                model.load_state_dict(model_loader)
                # Switch model to evaluation mode
                model.eval()
                #Transform image
                list_agu = [A.Normalize()]
                transform = A.Compose(list_agu)

                img = cv2.resize(image,(256,256))
                transformed = transform(image=img)
                img = transformed["image"]

                img = img.transpose(2, 0, 1)
                img = torch.tensor(img).float()
                img = img.to(device)
                img = img.view(1, *img.size()).to(device)

                with torch.no_grad():
                    pred = model(img.float())
                
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = pred.cpu().detach().numpy()
                accuracy_5.append(pred)

        st.success('Done!!!')
        return accuracy_5

def heatmap(model_name,image,Cam=GradCAM):
    '''
    '''
    image_ori = image
    if model_name == 'Efficient_B0_256':
        model = BaseNetwork('efficientnet_b0')
        model.load_state_dict(torch.load(os.path.join('model',random.choice(os.listdir('model'))),map_location=torch.device('cpu')),strict=False)
        target_layers = [model.conv_head]
        cam_image, image_scale = back_heatmap(model,image,target_layers,Cam)
    
    return cam_image, image_ori, image_scale
        

def back_heatmap(model,image,target_layers,Cam):
    '''
    '''
    image = cv2.resize(image,(256,256))
    image_scale = image
    cam_algorithm = Cam #GradCAM, \
                                    #     ScoreCAM, \
                                    #     GradCAMPlusPlus, \
                                    #     AblationCAM, \
                                    #     XGradCAM, \
                                    #     EigenCAM, \
                                    #     EigenGradCAM, \
                                    #     LayerCAM, \
                                    #     FullGrad
    image = image[:, :, ::-1]
    image = np.float32(image) / 255
    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=True) as cam:
            cam.batch_size = 30
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=None,
                                aug_smooth=False,
                                eigen_smooth=False)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return cam_image, image_scale


if __name__ == '__main__':
    load_result('Efficient_B0_256')

