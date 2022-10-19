#import librariesimport numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.xception import decode_predictions
import requests
import numpy as np

#load pre trained Xception model
model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=True)

#loading and preprocessing cat image
IMAGE_PATH='lion.jpg'
img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(299,299,3))
img=tf.keras.preprocessing.image.img_to_array(img)#view the image
#view the image
plt.imshow(img/255.)
#plt.show()


#fetching labels from Imagenet  
response=requests.get('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
imgnet_map=response.json()
imgnet_map={v[1]:k for k, v in imgnet_map.items()}#make model predictions
img=tf.keras.applications.xception.preprocess_input(img)
predictions=model.predict(np.array([img]))
print(decode_predictions(predictions,top=1))
