import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Reshape, Softmax, Conv2DTranspose, InputLayer, Cropping2D, Resizing
from keras.regularizers import l2

model = tf.keras.applications.MobileNet()

input_shape = (512, 512, 1)

# input_1 (InputLayer)                              (None, 224, 224, 3 )    0       <- remove
#                                                                                   <- insert conv2d + cropping
# conv1 (Conv2D)                                    (None, 112, 112, 32)    864                                                                           
# conv1_bn (BatchNormalizatio                       (None, 112, 112, 32)    128         

# ...

# conv_pw_13_bn (BatchNormalization)                (None, 7, 7, 1024)      4096      
# conv_pw_13_relu (ReLU)                            (None, 7, 7, 1024)      0         
# global_average_pooling2d (GlobalAveragePooling2D) (None, 1, 1, 1024)      0       <- remove                 
# dropout (Dropout)                                 (None, 1, 1, 1024)      0       <- remove  
# conv_preds (Conv2D)                               (None, 1, 1, 1000)      1025000 <- remove  
# reshape_2 (Reshape)                               (None, 1000)            0       <- remove  
# predictions (Activation)                          (None, 1000)            0       <- remove  
#                                                                                   <- insert conv2dtranspose + softmax

layers = []
layers.append(InputLayer(input_shape))
layers.append(Resizing(224, 224, "bicubic"))
layers.append(Conv2D(3, 8, padding="same"))
layers.extend(model.layers[1:-5])
layers.append(Reshape((32, 32, 49)))
layers.append(Conv2D(11, 8, padding="same"))
layers.append(Softmax(3))

model = Sequential(layers)