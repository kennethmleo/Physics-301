import numpy as np
from tensorflow.keras import applications
from tensorflow.keras import preprocessing
model = applications.mobilenet_v2.MobileNetV2(weights='imagenet')

def convert_size(img):
    centerx, centery = [int(img.shape[0] / 2) - 112, int(img.shape[1] / 2) - 112]
    img = img[centerx:centerx + 224, centery:centery + 224]
    return img

def Object_detection_mobilenetv2(image):
    if type(image) == str: 
        img = preprocessing.image.load_img(image, target_size=(224, 224))
        img = preprocessing.image.img_to_array(img)
    else: 
        img = convert_size(image)
    
    data = np.empty((1,224,224,3))
    data[0] = img
    data = applications.mobilenet_v2.preprocess_input(data)
    
    predictions = model.predict(data)
    
    class_ = []
    percent_ = []
    for name, desc, score in applications.mobilenet_v2.decode_predictions(predictions)[0]:
        class_.append(desc)
        percent_.append(100 * score)
    return class_, percent_