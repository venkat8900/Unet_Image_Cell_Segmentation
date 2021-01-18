# Unet network in keras
# data taken from: https://www.kaggle.com/c/data-science-bowl-2018/data

# import libraries
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm # progress bar during loop execution 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
import random

seed = 42
np.random.seed = seed # makes sure no matter how many times we run we get the same random number

# input image: 128 x 128 x 3
# define image dimensions
img_width = 128
img_height = 128
num_channels = 3 # RGB image

train_path = 'dataset/stage1_train/' # add training path
test_path = 'dataset/stage1_test/' # add testing path

# read all the id's for train images
train_ids = next(os.walk(train_path))[1] # gives the list of all train folder names
test_ids = next(os.walk(test_path))[1] # gives the list of all test folder names

# for training images
X_train = np.zeros((len(train_ids), img_height, img_width, num_channels), dtype = np.uint8)
Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype = np.bool) 


print('[INFO] Resizing train images and masks.... ')
for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):
	path = train_path + id_
	img = imread(path + '/images/' + id_ + '.png')[:,:,:num_channels] 
	img = resize (img, (img_height, img_width), mode = 'constant', preserve_range = True) # resize to input dimensions
	X_train[n] = img # filling with the values form the image
	mask = np.zeros((img_height, img_width, 1), dtype = np.bool)
	for mask_file in next(os.walk(path + '/masks/'))[2]:
		mask_ = imread(path + '/masks/' + mask_file)
		mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode = 'constant',
										preserve_range = True), axis = -1)
		mask = np.maximum(mask, mask_)
		# pixels which are having a cell will have some value for the mask. 
		# every pixel is 0 or 1. If it is 1 it has a cell 

	Y_train[n] = mask # we need to train with this 


#for test images -- do not have masks
X_test = np.zeros((len(test_ids), img_height, img_width, num_channels), dtype = np.uint8)
sizes_test = []
print("[INFO] Resizing test images....")
for n, id_ in tqdm(enumerate(test_ids), total = len(test_ids)):
	path = test_path + id_
	img = imread(path + '/images/' + id_ + '.png')[:,:,:num_channels]
	sizes_test.append([img.shape[0], img.shape[1]])
	img = resize (img, (img_height, img_width), mode = 'constant', preserve_range = True) # resize to input dimensions
	X_test[n] = img # filling with the values form the image

print("[INFO] Resizing Done!!...")
	


# build the Unet model
inputs = tf.keras.layers.Input((img_width, img_height, num_channels))

# convert each input into floating point
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs) 


# can use statistical normal distribution too to initlaize the weights.
# he_normal - truncated normal distribution centered around 0 - Gaussian distribution
# can use orthogonal, truncated normal, random normal. 

# padding = same --> input image size same as output image
conv_1_1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(s)
drop_1 = tf.keras.layers.Dropout(0.1)(conv_1_1)
conv_1_2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(drop_1)
max_pool_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_1_2)

conv_2_1 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(max_pool_1)
drop_2 = tf.keras.layers.Dropout(0.1)(conv_2_1)
conv_2_2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(drop_2)
max_pool_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_2_2)

conv_3_1 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(max_pool_2)
drop_3 = tf.keras.layers.Dropout(0.1)(conv_3_1)
conv_3_2 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(drop_3)
max_pool_3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_3_2)

conv_4_1 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(max_pool_3)
drop_4 = tf.keras.layers.Dropout(0.1)(conv_4_1)
conv_4_2 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(drop_4)
max_pool_4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_4_2)

conv_5_1 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(max_pool_4)
drop_5 = tf.keras.layers.Dropout(0.1)(conv_5_1)
conv_5_2 = tf.keras.layers.Conv2D(512, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(drop_5)


up_6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv_5_2)
up_6 = tf.keras.layers.concatenate([up_6, conv_4_2])
conv_6_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_6)
drop_6 = tf.keras.layers.Dropout(0.2)(conv_6_1)
conv_6_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop_6)

up_7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_6_2)
up_7 = tf.keras.layers.concatenate([up_7, conv_3_2])
conv_7_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_7)
drop_7 = tf.keras.layers.Dropout(0.2)(conv_7_1)
conv_7_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop_7)

up_8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_7_2)
up_8 = tf.keras.layers.concatenate([up_8, conv_2_2])
conv_8_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_8)
drop_8 = tf.keras.layers.Dropout(0.1)(conv_8_1)
conv_8_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop_8)

up_9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_8_2)
up_9 = tf.keras.layers.concatenate([up_9, conv_1_2], axis=3)
conv_9_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_9)
drop_9 = tf.keras.layers.Dropout(0.1)(conv_9_1)
conv_9_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop_9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_9_2)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall', 'TruePostives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])
model.summary(line_length=None, positions=None, print_fn=None)


# checkpoints
check_pointer = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose = 1, save_best_only = True)

# tensorboard callbaks to look at various things
callbacks = [
		tf.keras.callbacks.EarlyStopping(patience = 2, montior = 'val_loss'),
		tf.keras.callbacks.TensorBoard(log_dir = 'logs'), checkpointer] # observes for 2 more epochs before updating to make sure that this stoppoint is the best

results = model.fit(X_train, Y_train, validation_split = 0.1, batch_size = 32, epochs = 25, callbacks = callbacks)


## testing after training
idx = random.randint(0, len(X_train))

# after model.fit, every pixel has a probability value(between 0 to 1)
train_pred = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose = 1)
val_pred = model.predict(X_train[:int(X_train.shape[0]*0.9):], verbose = 1) 
test_pred = model.predict(X_test, verbose = 1)

# applying a threshold
train_pred_t = (train_pred > 0.5).astype(np.uint8)
val_pred_t = (val_pred > 0.5).astype(np.uint8)
test_pred_t = (test_pred > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(train_pred_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(train__pred_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(val_pred_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(val_pred_t[ix]))
plt.show()