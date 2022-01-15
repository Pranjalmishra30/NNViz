import requests
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

os.system("python3 ml_server.py")

URI = 'http://127.0.0.1:5000'
st.set_option('deprecation.showPyplotGlobalUse', False)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


st.title('NNViz')

st.markdown('''
NNViz is a neural network visualiser. We will be able to visualise the outputs of all the nodes of all
the layers of the neural network for any given image. A 3 layer NN model is built using the MNIST dataset.

Click on **Show Predictions** to get started.
''')

max =0

if st.button('Show Predictions'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))

    st.markdown('## Input Image')
    st.image(image, width=150)
    st.caption("Test image from the MNIST dataset")

    st.markdown('## Neural Network Model running ...')

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))

        plt.figure(figsize=(32, 4))

        if layer == 2:  # output layer
            row = 1
            col = 10
        else:
            row = 2
            col = 16

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow((number * np.ones((8, 8, 3))).astype('float32'), cmap='binary')
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
                max = np.argmax(numbers)

        st.markdown('##### Layer: {}'.format(layer + 1))
        st.pyplot()
    
    st.caption("The brightest block is the predicted number")
    st.write()
    st.markdown("### The predicted number is **{}**".format(max))
    st.markdown('Tip: Click **show predictions** to run a different test image')
    st.write()



link = 'Check out the source code [here](https://github.com/Pranjalmishra30/MNIST_WebApp)'
# st.write(link,unsafe_allow_html=True)
