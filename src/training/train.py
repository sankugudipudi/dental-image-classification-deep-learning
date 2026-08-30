"""
Training Pipeline for Dental Image Classification
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.models.transfer_learning import build_transfer_model

def train_pipeline(epochs=30, batch_size=32, num_classes=5):
    print('[*] Initializing ResNet50 Transfer Learning Model...')
    model = build_transfer_model('resnet50', num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('best_dental_model.keras', save_best_only=True)
    ]

    print('[*] Model compiled. Ready for dataset training execution.')
    return model

if __name__ == '__main__':
    train_pipeline()