# Importing the required packages
import streamlit as st
import cv2      
import os,urllib
import numpy as np    
import tensorflow as tf
import time

def main():
    #print(cv2.__version__)
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        (None, 'Evaluate the model')
        )
        
    if selected_box == 'Evaluate the model':
        models()


def models():

    st.title('Denoise your image with deep learning models..')
        
        
    st.write('\n')
    
    choice=st.sidebar.selectbox("Choose how to load image",[None, "Browse Image"])
    
    if choice=="Browse Image":
      uploaded_file = st.sidebar.file_uploader("Choose a image file", type="png")

      if uploaded_file is not None:
      # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        gt = cv2.imdecode(file_bytes, 1)
        prediction_ui(gt)

    
def prediction_ui(gt):

    models_load_state=st.text('\n Loading models..')
    print(get_models())
    dncnn,dncnn_lite,ridnet,ridnet_lite=get_models()
    print("This is a final sample print.......................................................................................")

    models_load_state=st.text('\n Models Loading..complete')

    dncnn_filesize,dncnnlite_filesize=get_filesizes()
    
    noise_level = st.sidebar.slider("Pick the noise level", 0, 45, 0)
          
    ground_truth,noisy_image,patches_noisy=get_image(gt,noise_level=noise_level)
    st.header('Input Image')
    st.markdown('** Noise level : ** `%d`  ( Noise level `0` will be same as original image )'%(noise_level))
    st.image(noisy_image)

    gt=simplest_cb(gt)
    ground_truth,noisy_image,patches_noisy=get_image(gt,noise_level=noise_level)
    st.header('Simplest Color Balance Image')
    st.markdown('** Noise level : ** `%d`  ( Noise level `0` will be same as original image )'%(noise_level))
    st.image(noisy_image)

    gt=CLAHE(gt)
    ground_truth,noisy_image,patches_noisy=get_image(gt,noise_level=noise_level)
    st.header('CLAHE')
    st.markdown('** Noise level : ** `%d`  ( Noise level `0` will be same as original image )'%(noise_level))
    st.image(noisy_image)

    

    if noise_level!=0:
      st.success('PSNR of Noisy image : %.3f db'%PSNR(ground_truth,noisy_image))
      
    model = st.sidebar.radio("Choose a model to predict",('DNCNN', 'None'),0)
    
    
    submit = st.sidebar.button('Predict Now')
          
  
    if submit and noise_level>=0:
    
        if model=='DNCNN':
            progress_bar = st.progress(0)
            start=time.time()
            progress_bar.progress(10)
            denoised_image=predict_fun(dncnn,patches_noisy,gt)
            progress_bar.progress(40)
            end=time.time()
            st.header('Denoised image using DnCNN model')
            st.markdown('( Size of the model is : `%.3f` MB ) ( Time taken for prediction : `%.3f` seconds )'%(dncnn_filesize,(end-start)))
            st.image(denoised_image)     
            st.success('PSNR of denoised image : %.3f db  '%(PSNR(ground_truth,denoised_image)))
            
            progress_bar.progress(60)
            start=time.time()
            denoised_image_lite=predict_fun_tflite(dncnn_lite,patches_noisy,gt)
            end=time.time()
            st.header('Denoised image using lite version of DnCNN model with unsharp masking')
            st.markdown('( Size of the model is : `%.3f` MB ) ( Time taken for prediction : `%.3f` seconds )'%(dncnnlite_filesize,(end-start)))
            progress_bar.progress(90)
            st.image(denoised_image_lite)
            st.success('PSNR of denoised image : %.3f db  '%(PSNR(ground_truth,denoised_image_lite)))
            progress_bar.progress(100)
            progress_bar.empty()
            
            
        
        elif model=='RIDNET':
            pass
            
                 

    elif submit==True and noise_level<10:
        st.error("Choose a minimum noise level of 10 ...")


def get_models():
    dncnn=tf.keras.models.load_model('dncnn.h5')
    ridnet=tf.keras.models.load_model('ridnet.h5')
    dncnn_lite = tf.lite.Interpreter('dncnn2.tflite')
    dncnn_lite.allocate_tensors()


    ridnet_lite = tf.lite.Interpreter('ridnet.tflite')
    ridnet_lite.allocate_tensors()
    return (dncnn,dncnn_lite,ridnet,ridnet_lite)
  
@st.cache
def get_filesizes():
    
    dncnn_filesize=os.stat('dncnn.h5').st_size / (1024 * 1024)
    dncnnlite_filesize=os.stat('dncnn2.tflite').st_size / (1024 * 1024)
    ridnet_filesize=os.stat('ridnet.h5').st_size / (1024 * 1024)
    ridnetlite_filesize=os.stat('ridnet.tflite').st_size / (1024 * 1024)
    return dncnn_filesize,dncnnlite_filesize
        
def get_patches(image):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height, width , channels= image.shape
    crop_sizes=[1]
    patch_size=40
    patches = []
    for crop_size in crop_sizes: #We will crop the image to different sizes
        crop_h, crop_w = int(height*crop_size),int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h-patch_size+1, int(patch_size/1)):
            for j in range(0, crop_w-patch_size+1, int(patch_size/1)):
              x = image_scaled[i:i+patch_size, j:j+patch_size] # This gets the patch from the original image with size patch_size x patch_size
              patches.append(x)
    return patches



def create_image_from_patches(patches,image_shape):
  '''This function takes the patches of images and reconstructs the image'''
  image=np.zeros(image_shape) # Create a image with all zeros with desired image shape
  patch_size=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_size+1,int(patch_size/1)):
    for j in range(0,image.shape[1]-patch_size+1,int(patch_size/1)):
      image[i:i+patch_size,j:j+patch_size]=patches[p] # Assigning values of pixels from patches to image
      p+=1
  return np.array(image)

def get_image(gt,noise_level):
  print("work")
  patches=get_patches(gt)
  height, width , channels= gt.shape
  test_image=cv2.resize(gt, (width//40*40,height//40*40), interpolation=cv2.INTER_CUBIC)
  patches=np.array(patches)
  ground_truth=create_image_from_patches(patches,test_image.shape)
  
  #predicting the output on the patches of test image
  patches = patches.astype('float32') /255.
  patches_noisy = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
  patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
  noisy_image=create_image_from_patches(patches_noisy,test_image.shape)
  
  return ground_truth/255.,noisy_image,patches_noisy
def predict_fun(model,patches_noisy,gt):

  height, width , channels= gt.shape
  gt=cv2.resize(gt, (width//40*40,height//40*40), interpolation=cv2.INTER_CUBIC)
  denoised_patches=model.predict(patches_noisy)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image=create_image_from_patches(denoised_patches,gt.shape)
  return denoised_image
  

def predict_fun_tflite(model,patches_noisy,gt):
    
  height, width , channels= gt.shape
  gt=cv2.resize(gt, (width//40*40,height//40*40), interpolation=cv2.INTER_CUBIC)
  
  denoised_patches=[]
  for p in patches_noisy:
    model.set_tensor(model.get_input_details()[0]['index'],tf.expand_dims(p,axis=0))
    model.invoke()
    pred=model.get_tensor(model.get_output_details()[0]['index'])
    pred=tf.squeeze(pred,axis=0)
    denoised_patches.append(pred)
  
  denoised_patches=np.array(denoised_patches)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image=create_image_from_patches(denoised_patches,gt.shape)

  return denoised_image  
  
def PSNR(gt, image, max_value=1):
    """"Function to calculate peak signal-to-noise ratio (PSNR) between two images."""
    height, width , channels= gt.shape
    gt=cv2.resize(gt, (width//40*40,height//40*40), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((gt - image) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def unsharpMasking(out):
    image = out
    
    a = (-1/256)
    kernel = a*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    print(image[1][0:5],image_sharp[1][0:5])
    cv2.imwrite("a.jpg", image_sharp)
    st.image("a.jpg")
    return image_sharp

def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 50.0,
        img.shape[0] * img.shape[1] * (1 - percent / 50.0)
    )
    for channel in cv2.split(img):
        
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

def CLAHE(img2):
    lab= cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl,a,b))

    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)    
    return enhanced_img

if __name__ == "__main__":
    main()
