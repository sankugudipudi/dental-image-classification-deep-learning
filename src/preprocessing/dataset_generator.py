"""
Keras Dataset Generator with Real-Time Medical Augmentation
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return train_datagen, test_datagen