import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from PIL import Image
import streamlit as st
"""
lr =0.8
size = 224
iterations = 40
style_wt = 1e5
content_wt = 1.0
"""
def show_process(s):
	st.write(s)

def display_plot(c,i):
	fig = plt.figure()
	plt.plot(range(i), c)
	plt.xlabel("Iterations")
	plt.ylabel("Total Cost")
	st.pyplot(fig)




def start_process(lr,size,iterations,style_wt,content_wt,content_image_path,style_image_path):
	#content_image_path = r"C:\Users\Asus\Desktop\NST\SONATA-hero-option1-764A5360-edit.jpg"
	#style_image_path = r"C:\Users\Asus\Desktop\NST\dfdsafdsrf.jpg"

	#print("TensorFlow version:", tf.__version__)

	style_layer_wts = [1.0, 0.8, 0.1, 0.1, 0.2]

	model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(size, size, 3))
	model.trainable = False

	def preprocess_image(image_path):
	    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(size, size))
	    img = tf.keras.preprocessing.image.img_to_array(img)
	    img = tf.keras.applications.vgg19.preprocess_input(img)
	    return np.expand_dims(img, axis = 0)

	def deprocess(x):
	    x[:, :, 0] += 103.939
	    x[:, :, 1] += 116.779
	    x[:, :, 2] += 123.68
	    x = x[:, :, ::-1]

	    x = np.clip(x, 0, 255).astype('uint8')
	    return x

	def display_image(image):
	    if len(image.shape) == 4:
	        image = image[0,:,:,:]

	    img = deprocess(image)

	content_layer = 'block4_conv2'

	content_model = tf.keras.models.Model(
	    inputs=model.input,
	    outputs=model.get_layer(content_layer).output
	)
	style_layers = [
	    'block1_conv1', 'block1_conv2',
	    'block2_conv1', 'block3_conv2',
	    'block5_conv1'
	    ]

	style_models = [
	    tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer).output)
	    for layer in style_layers
	]
	def content_cost(content_img, generated_img):
	    C = content_model(content_img)
	    G = content_model(generated_img)
	    cost = tf.reduce_mean(tf.square(C - G))
	    return cost
	def gram_matrix(M):
	    num_channels = tf.shape(M)[-1]
	    M = tf.reshape(M, shape=(-1, num_channels))
	    n = tf.shape(M)[0]
	    G = tf.matmul(tf.transpose(M), M)
	    return G / tf.cast(n, dtype=tf.float32)

	def style_cost(style_img, generated_img):
	    total_cost = 0
	    
	    for i, style_model in enumerate(style_models):
	        S = style_model(style_img)
	        G = style_model(generated_img)
	        GS = gram_matrix(S)
	        GG = gram_matrix(G)
	        current_cost = style_layer_wts[i] * tf.reduce_mean(tf.square(GS - GG))
	        total_cost += current_cost
	    
	    total_cost /= (size * size * len(style_models))
	    return total_cost

	content_image_preprocessed = preprocess_image(content_image_path)
	style_image_preprocessed = preprocess_image(style_image_path)
	generated_image = tf.Variable(content_image_preprocessed, dtype=tf.float32)

	generated_images = []
	costs = []

	optimizer = tf.optimizers.Adam(learning_rate=lr)


	for i in range(iterations):
	    
	    with tf.GradientTape() as tape:
	        J_content = content_cost(content_img=content_image_preprocessed, generated_img=generated_image)
	        J_style = style_cost(style_img=style_image_preprocessed, generated_img=generated_image)
	        J_total = content_wt * J_content + style_wt * J_style
	    
	    gradients = tape.gradient(J_total, generated_image)
	    optimizer.apply_gradients([(gradients, generated_image)])
	    
	    costs.append(J_total.numpy())
	    
	    if i % 2 == 0:
	        display_image(generated_image.numpy())
	        generated_images.append(generated_image.numpy())
	        detail = "Iteration:{}/{}, Total Cost:{}, Style Cost: {}, Content Cost: {}".format(i+1, iterations, J_total, J_style, J_content)
	        show_process(detail)



	image = Image.fromarray(deprocess(generated_images[-1][0]))
	plt.imshow(image)
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	plt.savefig('out.png')
	out_path = r"C:\Users\Asus\MLapp\out.png"


	display_plot(costs,iterations)
	return out_path


















