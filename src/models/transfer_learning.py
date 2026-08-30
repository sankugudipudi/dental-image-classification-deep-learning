"""
Transfer Learning Models: ResNet50 and EfficientNetB0 Backbones
"""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0

def build_transfer_model(backbone_name='resnet50', input_shape=(224, 224, 3), num_classes=5) -> models.Model:
    if backbone_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = False  # Freeze initial feature extractor

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)