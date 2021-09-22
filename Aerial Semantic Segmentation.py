# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:22:32 2021

@author: ahmed
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import os
import gc
from tqdm import tqdm

import cv2
import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Dropout, concatenate, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import segmentation_models as sm


scaler = MinMaxScaler()

# Loading Data

original_images_path = "E:/Software/professional practice projects/In progress/dataset/semantic_drone_dataset/original_images"
original_labels_path = "E:/Software/professional practice projects/In progress/dataset/semantic_drone_dataset/label_images_semantic"
rgb_color_masks_path = "E:/Software/professional practice projects/In progress/RGB_color_image_masks/RGB_color_image_masks"


images_list = []

for img_name in tqdm(os.listdir(original_images_path)):
    img_path = os.path.join(original_images_path, img_name)
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    
    images_list.append(img)



masks_list = []

for mask_name in tqdm(os.listdir(original_labels_path)):
    mask_path = os.path.join(original_labels_path, mask_name)
    mask = cv2.imread(mask_path, 1)
    mask = cv2.resize(mask, (256, 256))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    masks_list.append(mask)



rgb_mask_list = []

for mask_name in tqdm(os.listdir(rgb_color_masks_path)):
    mask_path = os.path.join(rgb_color_masks_path, mask_name)
    mask = cv2.imread(mask_path, 1)
    mask = cv2.resize(mask, (256, 256))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
 
    rgb_mask_list.append(mask)
    

gc.collect()



# Visualizing data

plt.figure(figsize=(10, 11))
idx=0

for i in range(3):
    rand_idx = random.randint(0, len(images_list))
    
    idx += 1
    plt.subplot(3, 3, idx)
    img = images_list[rand_idx]
    plt.imshow(img)
    plt.title("Original Image")
    
    idx += 1
    plt.subplot(3, 3, idx)
    mask = masks_list[rand_idx]
    plt.imshow(mask)
    plt.title("True Mask")
    
    idx += 1
    plt.subplot(3, 3, idx)
    rgb_mask = rgb_mask_list[rand_idx]
    plt.imshow(rgb_mask)
    plt.title("True RGB Mask")




# No need for all three channels so i'll take only one channel
for i in range(len(masks_list)):
    masks_list[i] = masks_list[i][:,:,0]

# for i in range(len(rgb_mask_list)):
#     rgb_mask_list[i] = rgb_mask_list[i][:,:,0]




class_dict_seg_path = "E:/Software/professional practice projects/In progress/class_dict_seg.csv"

class_dict_seg = pd.read_csv(class_dict_seg_path)
class_dict_seg.info()
class_dict_seg.head()



images_list = np.array(images_list)
masks_list = np.array(masks_list)
rgb_mask_list = np.array(rgb_mask_list)


print(np.unique(masks_list))

num_classes = len(np.unique(masks_list))

masks_list = to_categorical(masks_list)



# Splitting data
x_train, x_val, y_train, y_val = train_test_split(images_list, masks_list, test_size=0.15)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)


################ Creating UNet Model ##########################
""" Encoder """



def conv2D_block(input_tensor, n_filters, kernel_size=3):
    
    x = input_tensor
    
    for i in range(2):
    
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same')(x)
        
        x = Activation('relu')(x)
    
    return x




def encoder_block(inputs, n_filters, pool_size, dropout):
    
    conv_output = conv2D_block(inputs, n_filters = n_filters)
    
    x = MaxPooling2D(pool_size=pool_size)(conv_output)
    
    x = Dropout(dropout)(x)
    
    return conv_output, x


def Encoder(inputs):
    
    conv_out_1, x1 = encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3)
    
    conv_out_2, x2 = encoder_block(x1, n_filters=128, pool_size=(2, 2), dropout=0.3)
    
    conv_out_3, x3 = encoder_block(x2, n_filters=128, pool_size=(2, 2), dropout=0.3)
    
    conv_out_4, x4 = encoder_block(x3, n_filters=512, pool_size=(2, 2), dropout=0.3)
    
    return x4, (conv_out_1, conv_out_2, conv_out_3, conv_out_4)



def Bottleneck(inputs):
    
    bottle_neck = conv2D_block(inputs, n_filters=1024)
    
    return bottle_neck




""" Decoder"""

def decoder_block(inputs, conv_output, n_filters, kernel_size, strides, dropout):
    
    trans = Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    
    conct = concatenate([trans, conv_output])
    
    x = Dropout(dropout)(conct)
    
    x = conv2D_block(x, n_filters, kernel_size=3)
    
    return x




def Decoder(inputs, convs, num_classes):
    
    conv_1, conv_2, conv_3, conv_4 = convs
    
    x1 = decoder_block(inputs, conv_4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    x2 = decoder_block(x1, conv_3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    x3 = decoder_block(x2, conv_2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    x4 = decoder_block(x3, conv_1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    outputs = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax', padding='same')(x4)
    
    return outputs





def UNet(num_classes):
    
    inputs = Input(shape=(256, 256, 3))
    
    encoder_output, convs = Encoder(inputs)
    
    bottle_neck = Bottleneck(encoder_output)
    
    outputs = Decoder(bottle_neck, convs, num_classes)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


model = UNet(num_classes)

model.summary()



my_callbacks = [ModelCheckpoint(filepath="model.h5", monitor='val_loss', verbose=1, save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
                CSVLogger("train_performance_per_epoch.csv")]


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, 
                    callbacks=my_callbacks, validation_data=(x_val, y_val))




y_pred = model.predict(x_test)

y_pred_argmx = np.argmax(y_pred, axis=3)

y_test_argmx = np.argmax(y_test, axis=3)

M_IoU = MeanIoU(num_classes=num_classes)

M_IoU.update_state(y_true=y_test_argmx, y_pred=y_pred_argmx)

print(M_IoU.result().numpy())

model.save('Aerial_Semantic_Segmentation.h5')
model.save_weights('Aerial_Semantic_Segmentation_weights.h5')

json_model = model.to_json()
with open("E:/Software/professional practice projects/In progress/Aerial_Semantic_Segmentation.json", 'w') as json_file:
    json_file.write(json_model)
    


plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
plt.title("Aerial_Semantic_Segmentation Accuracy")


plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['loss', 'val_loss'])
plt.title("Aerial_Semantic_Segmentation Loss")




plt.figure(figsize=(10, 11))
idx=0

for i in range(3):
    rand_idx = random.randint(0, len(x_test))
    
    idx += 1
    plt.subplot(3, 3, idx)
    img = x_test[rand_idx]
    plt.imshow(img)
    plt.title("Original Image")
    
    idx += 1
    plt.subplot(3, 3, idx)
    mask = y_test_argmx[rand_idx]
    plt.imshow(mask)
    plt.title("True Mask")
    
    idx += 1
    plt.subplot(3, 3, idx)
    rgb_mask = y_pred_argmx[rand_idx]
    plt.imshow(rgb_mask)
    plt.title("Pred Mask")




# BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)

# # preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)
# x_test = preprocess_input(x_test)

# # define model
# model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=num_classes, activation='softmax')

# model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print(model_resnet_backbone.summary())


# history2=model_resnet_backbone.fit(x_train, 
#           y_train,
#           batch_size=16, 
#           epochs=100,
#           verbose=1,
#           validation_data=(x_val, y_val))


























