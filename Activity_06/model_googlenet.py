import numpy as np
from tensorflow.keras import applications
from tensorflow.keras import preprocessing
model = applications.inception_v3.InceptionV3(weights='imagenet')

def convert_size_googlenet(img):
    centerx, centery = [int(img.shape[0] / 2) - 299//2, int(img.shape[1] / 2) - 299//2]
    img = img[centerx:centerx + 299, centery:centery + 299]
    return img

def Object_detection_googlenet(image):
    if type(image) == str: 
        img = preprocessing.image.load_img(image, target_size=(299, 299))
        img = preprocessing.image.img_to_array(img)
    else: 
        img = convert_size_googlenet(image)
    
    data = np.empty((1,299,299,3))
    data[0] = img
    data = applications.inception_v3.preprocess_input(data)
    
    predictions = model.predict(data)
    
    class_ = []
    percent_ = []
    for name, desc, score in applications.inception_v3.decode_predictions(predictions)[0]:
        class_.append(desc)
        percent_.append(100 * score)
    return class_, percent_