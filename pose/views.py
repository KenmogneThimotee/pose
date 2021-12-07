from django.shortcuts import render

# Create your views here.
# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from .forms import UploadFileForm
from rest_framework.response import Response
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.decorators import api_view
from rest_framework import status
from django.conf import settings


def comparator(image):

    #image = tf.convert_to_tensor(image)

    #image = tf.compat.v1.image.decode_jpeg(image)
    #image = tf.compat.v1.image.decode_jpeg(image)
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

    keypoints = np.squeeze(keypoints)[:,:2]
    
    return keypoints

def normalize(image):
    image *= 255.0/image.max()

    return image

def cosineSimilarity(inputs, source):

        inputs = normalize(inputs.copy())
        source = normalize(source.copy())

        print("inputs")
        print(inputs.shape)

        print("source")
        print(source.shape)

        similarity = cosine_similarity(inputs.reshape(1,-1), source.reshape(1,-1)) + 1

        similarity = similarity / 2

        print("Similarity test size")
        print(similarity)

        return similarity * 100


@api_view(['POST'])
def comparePose(request):

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():

            fl = request.FILES['file']
            if fl.size > 3000000:
                return Response({"message":"Taille du fichier superieur a 3mb"},
                status=status.HTTP_406_NOT_ACCEPTABLE)
            else:
                img_str = fl.read()
                nparr = np.fromstring(img_str, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                input_new_coords = comparator(img_np)

                
                try:
                    saved_profile = np.load(settings.PATH_SOURCE)
                except:
                    return Response({"message":"Set source image first"}, status=status.HTTP_406_NOT_ACCEPTABLE)

                score = cosineSimilarity(input_new_coords, saved_profile)

                if score >= 99:
                    return Response({"message":"same"}, status=status.HTTP_200_OK)
                else:
                    return Response({"message":"diff"}, status=status.HTTP_200_OK)
        else:
            return Response({"message":" unsuccess"}, status=status.HTTP_406_NOT_ACCEPTABLE)


@api_view(['POST'])
def setImageSource(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            fl = request.FILES['file']
            if fl.size > 3000000:
                return Response({"message":"Taille du fichier superieur a 3mb"}, status=status.HTTP_406_NOT_ACCEPTABLE)
            else:
                img_str = fl.read()
                nparr = np.fromstring(img_str, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                #prediction
                keypoints = comparator(img_np)
                np.save(settings.PATH_SOURCE, keypoints)

            return Response({"message":"success"}, status=status.HTTP_200_OK)
        else:
            return Response({"message":" unsuccess"}, status=status.HTTP_200_OK)






