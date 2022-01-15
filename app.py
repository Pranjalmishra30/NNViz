import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

def GetPredictions():
    model = tf.keras.models.load_model('model/Basic_model.h5')
    feature_model = tf.keras.models.Model(model.inputs, [layer.output for layer in model.layers]) # 2nd model created to output hidden layers

    _, (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.

    index = np.random.choice(x_test.shape[0])
    image = x_test[index,:,:]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image

st.set_page_config(page_title='NNViz', page_icon='🤖')
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
NNViz is a neural network visualiser. We will be able to visualise the outputs of all the layers of the neural network for any input image. Here a 3 layer NN model is built using the MNIST dataset.

Click on **Show Predictions** to get started.
''')

max =0

if st.button('Show Predictions 👇'):
    preds,image = GetPredictions()
    image = np.reshape(image, (28, 28))

    st.markdown('### Input Image')
    st.image(image, width=150)
    st.caption("Test image from the MNIST dataset")

    st.markdown('### Neural Network building ...')

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



link = 'Check out the project [here](https://github.com/Pranjalmishra30/NN_Viz)'
st.write(link,unsafe_allow_html=True)
