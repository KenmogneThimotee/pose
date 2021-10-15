from django.shortcuts import render

# Create your views here.
# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from .forms import UploadFileForm
from rest_framework.response import Response
from django.conf import settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity






def comparator(image_1):

    image = tf.convert_to_tensor(image_1)

    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    # Download the model from TF Hub.
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    # Run model inference.
    outputs = movenet(image)

    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    keypoints = np.squeeze(keypoints)[:,:2]
    
    return keypoints

def normalize(input_test):
	for k in range(0,17):	
		input_test[:,k] = input_test[:,k]/np.linalg.norm(input_test[:,k])	
	return input_test

def cosineSimilarity(self, inputs, source):

		inputs = normalize(inputs)
		source = normalize(source)

		similarity = cosine_similarity(inputs, source) + 1

		similarity = similarity / 2

		return similarity * 100


def comparePose(request):

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():

            fl = request.FILES['file']
            if fl.size > 3000000:
                return Response({"message":"Taille du fichier superieur a 3mb"})
            else:
                img_str = fl.read()
                nparr = np.fromstring(img_str, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
                input_new_coords = comparator(img_np)

                saved_profile = np.load(settings.PATH_SOURCE)

                score = score = cosineSimilarity(input_new_coords, saved_profile)

                if score >= 70:
                    return Response({"message":"same"})
                else:
                    return Response({"message":"diff"})
        else:
            return Response({"message":" unsuccess"})

def setImageSource(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            fl = request.FILES['file']
            if fl.size > 3000000:
                return Response({"message":"Taille du fichier superieur a 3mb"})
            else:
                img_str = fl.read()
                nparr = np.fromstring(img_str, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
                #prediction
                keypoints = comparator(img_np)
                np.save(settings.PATH_SOURCE, keypoints)

            return Response({"message":"success"})
        else:
            return Response({"message":" unsuccess"})






