import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_seleection import train_test_split
from sklearn.linear_model import LogsiticRegression
from PIL import Image
import PIL.ImageOps

X,Y = fetch_openml('mnist_784', version = 1, return_X_y = True)

xtrain, xtest, ytrain, ytest = train_test_split(X,y, random_state = 9, train_size = 7500, test_size = 2500)
xtrain_scaled = xtrain/255.00
xtest_scaled = xtest/255.00

clf = LogsiticRegression(solver = 'saga', multi_class = 'multinomial').fit(xtrain_scaled, ytrain)

def get_prediction(Image):
    im_pil = Image.open(Image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS())
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_ivertedScaled = np.clip(image_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_ivertedScaled = np.asarray(image_bw_resized_ivertedScaled)/max_pixel
    testsample = np.array(image_bw_resized_ivertedScaled).reshape(1, 784)
    testpred = clf.predict(testsample)
    return testpred[0]

    