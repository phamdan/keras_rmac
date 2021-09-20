from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import keras
from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map
import scipy.io
import numpy as np
import utils
########################

from keras import backend as K
import keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras import Model, layers
from keras.preprocessing import image
from keras.models import load_model, model_from_json
import PIL
import os
# from keras import backend as K
# K.set_image_dim_ordering('th') 
########################

def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):

    # Load VGG16

    vgg16_model = VGG16(utils.DATA_DIR + utils.WEIGHTS_FILE, input_shape)
  
    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling

    x = RoiPooling([1], num_rois)([vgg16_model.layers[-1].output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([vgg16_model.input, in_roi], rmac_norm)

    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])

    return model
def preprocessImageAndGetInput2(linkImage):
    
    img = image.load_img(linkImage)

    # Resize
    scale = utils.IMG_SIZE / max(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
#     print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size)

    # Mean substraction
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_image(x)
    return x
def get_vector(x): #params tensor image 
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
    regions = rmac_regions(Wmap, Hmap, 3)
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
    # Compute RMAC vector
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    return RMAC

if __name__ == "__main__":
    i=0
    dataset= os.listdir("query/")
    from datetime import datetime
    t1 = datetime.now().time()
    vector_dataset=[]
    for img in dataset:
        print(i)
        i+=1
        file = "query/"+img
        x=preprocessImageAndGetInput2(file)
        RMAC=get_vector(x)
        vector_query.append(RMAC)
    t2 = datetime.now().time()
    