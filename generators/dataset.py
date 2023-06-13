import os
import tensorflow as tf
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
TEST_DIR = 'test'
TRAIN_N_DIR = os.path.join(TRAIN_DIR, 'n')
TRAIN_Y_DIR = os.path.join(TRAIN_DIR, 'y')
VALIDATION_N_DIR = os.path.join(VALIDATION_DIR, 'n')
VALIDATION_Y_DIR = os.path.join(VALIDATION_DIR, 'y')
TEST_N_DIR = os.path.join(TEST_DIR, 'n')
TEST_Y_DIR = os.path.join(TEST_DIR, 'y')
DATASETS = {
    'normal': os.path.join(BASE_DIR, "processed_data"),
    'canny': os.path.join(BASE_DIR, "processed_data_canny"),
    'opencv': os.path.join(BASE_DIR, "processed_data_opencv"),
}

def get_train(dataset):
    color_mode = 'grayscale'
    if dataset == 'normal':
        color_mode = 'rgb'
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASETS[dataset], TRAIN_DIR),
        target_size = (700, 700),
        batch_size = 200*2,
	    shuffle = False,
        seed = 6,
        class_mode = 'binary',
        color_mode = color_mode
    )
    return train_generator

def get_val(dataset):
    color_mode = 'grayscale'
    if dataset == 'normal':
        color_mode = 'rgb'
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATASETS[dataset], VALIDATION_DIR),
        target_size = (700,700),
        batch_size = 50*2,
	    shuffle = False,
        seed = 6,
        class_mode = 'binary',
        color_mode = color_mode
    )
    return validation_generator

def get_test(dataset):
    color_mode = 'grayscale'
    if dataset == 'normal':
        color_mode = 'rgb'
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASETS[dataset], TEST_DIR),
        target_size = (700,700),
        batch_size = 50*2,
    	shuffle = False,
        seed = 6,
        class_mode = 'binary',
        color_mode = color_mode
    )
    return test_generator
