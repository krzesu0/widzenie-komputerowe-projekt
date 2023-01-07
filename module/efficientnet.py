import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Reshape, Softmax, Conv2DTranspose, Resizing, UpSampling2D
from keras.regularizers import l2

model = tf.keras.applications.EfficientNetB0()

input_shape = (512, 512, 1)

# input_1 (InputLayer)                              (None, 224, 224, 3)     0       []                      <- remove
# rescaling (Rescaling)                             (None, 224, 224, 3)     0       ['input_1[0][0]']       <- remove
# normalization (Normalization)                     (None, 224, 224, 3)     7       ['rescaling[0][0]']     
# rescaling_1 (Rescaling)                           (None, 224, 224, 3)     0       ['normalization[0][0]'] 
#
# ...
#
# block7a_project_bn (BatchNormalization)           (None, 7, 7, 320)       1280    ['block7a_project_conv[0][0]']
# top_conv (Conv2D)                                 (None, 7, 7, 1280)      409600  ['block7a_project_bn[0][0]']
# top_bn (BatchNormalization)                       (None, 7, 7, 1280)      5120    ['top_conv[0][0]']
# top_activation (Activation)                       (None, 7, 7, 1280)      0       ['top_bn[0][0]']
# avg_pool (GlobalAveragePooling2D)                 (None, 1280)            0       ['top_activation[0][0]']    <- remove    
# top_dropout (Dropout)                             (None, 1280)            0       ['avg_pool[0][0]']          <- remove    
# predictions (Dense)                               (None, 1000)            1281000 ['top_dropout[0][0]']       <- remove                                                                      <- insert conv2dtranspose + softmax

# layers.append(InputLayer(input_shape))
# layers.append(Resizing(224, 224, "bicubic"))
# layers.append(Conv2D(3, 8, padding="same", input_shape=input_shape))
# layers.extend(model.layers[1:-5])
# layers.append(Reshape((32, 32, 49)))
# layers.append(Conv2D(11, 8, padding="same", input_shape=input_shape))
# layers.append(Softmax(3))

model = Model(inputs=model.inputs, outputs=model.layers[-4].output)

input_layer = tf.keras.Input(input_shape)
resize      = Resizing(224, 224, "bicubic")(input_layer)
add_chann   = Conv2D(3, 8, padding="same")(resize)
rest        = model(add_chann)
reshape     = Reshape((16, 16, 245))(rest)
upsample    = UpSampling2D((2, 2))(reshape)
downscale   = Conv2D(11, 4, padding="same")(upsample)
softmax     = Softmax(3)(downscale)
 
model = Model(inputs=input_layer, outputs=softmax)

# model = Sequential(layers)