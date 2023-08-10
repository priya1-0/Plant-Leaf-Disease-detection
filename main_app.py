#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('C:/Users/Lenovo/Desktop/my_computer/1st year/2nd semester/MINIPROJECT/final/Plant Disease/Plant_Disease/plant_disease.h5')

#Name of Classes
CLASS_NAMES = ['''Tomato-Bacterial_spot-A plant with bacterial spot cannot be cured.''', '''Tomato-Early_Blight- Use Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate''','''Tomato-Healthy-tip: more sun more fruit''']


# #Setting Title of App
# st.title("Plant Disease Detection")
# st.markdown("Upload an image of the plant leaf")
#Setting Title of App
#st.title("")
st.markdown('<p style="font-family:Georgia, serif; text-align: center; color:white; font-size: 60px;">Plant Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p style="font-family:Georgia, serif; color:black; font-size: 30px;">Upload an image of the plant leaf</p>', unsafe_allow_html=True)
#st.markdown("Upload an image of the plant leaf")

#Uploading the image
plant_image = st.file_uploader("", type="jpg")


submit = st.button("Predict")
#st.sucess, st.warning, st.error and st.excep
#st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)
#background image
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1533119408463-b0f487583ff6?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=880&q=80");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# #Uploading the dog image
# plant_image = st.file_uploader("Choose an image...", type="jpg")
# submit = st.button('Predict')
#On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
        st.markdown(str("SOLUTION: " + result.split('-')[2]))
        # st.markdown(str("*" + result.split('-')[3]))
        # st.markdown(str("*" + result.split('-')[4]))
