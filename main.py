"""
Future Scope : Try to implement a feature to select models other than the current default of VGG19
"""




import streamlit as st 
from PIL import Image
from model import start_process
import os
import matplotlib.pyplot as plt


st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    text-align : center;
}
</style>
""", unsafe_allow_html=True)





st.markdown("""
<style>
body {
    color: #02261c;
    background-color: #b0ebbb;
}
</style>
    """, unsafe_allow_html=True)


st.markdown(
    """
<style>
.css-1aumxhk {
background-color: #b0ebea;
background-image: none;
color: #425e35
}
</style>
""",
    unsafe_allow_html=True,
)



st.sidebar.title("Select Parameters")

st.title("Neural Style Transfer")
filename1,filename2 = '',''
folder_path=os.getcwd() + r"\images"
filenames = os.listdir(folder_path)
if filename1=='' or filename2=='':
	selected_filename = st.sidebar.selectbox('Select a content image', filenames)
filename1 = os.path.join(folder_path, selected_filename) 
filenames = os.listdir(folder_path)
if filename1=='' or filename2=='':
	selected_filename = st.sidebar.selectbox('Select a style image', filenames)
filename2 = os.path.join(folder_path,selected_filename) 
print(filename1)
print(filename2)

if filename1!='' and filename2!='':
	lr = st.sidebar.slider('Learning Rate', 0.1, 1.0)
	#size = st.sidebar.slider('Size of Image', 200, 500)
	iterations = st.sidebar.slider('No of iterations', 2, 300)
	style_wt = st.sidebar.slider('Style Weight', 100, 200000)
	content_wt = st.sidebar.slider('Content Weight', 1, 100)
	image1 = Image.open(filename1)
	image2 = Image.open(filename2)
	st.image(image1, caption='', use_column_width=True)
	st.markdown('<p class="big-font">Content Image</p>', unsafe_allow_html=True)

	st.image(image2, caption='', use_column_width=True)
	st.markdown('<p class="big-font">Content Image</p>', unsafe_allow_html=True)

	st.write("")
	#st.write("Creating Your Image....")

	if st.sidebar.button('Start') == True:
		st.write("Creating Your Image....")


		label = start_process(lr,size,iterations,style_wt,content_wt,filename1,filename2)
		out_image = Image.open(label)
		st.image(out_image, caption='Output Image', use_column_width=True)

	

    