# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# # Define paths
# train_dir = '../dataset/train'
# model_path = '../models/saved_model/deepfake_model.h5'

# # Image parameters
# IMG_HEIGHT, IMG_WIDTH = 224, 224
# BATCH_SIZE =128
# EPOCHS = 3

# # Data generators with augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real']
# )

# # Build CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train model
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE
# )

# # Save model
# os.makedirs(os.path.dirname(model_path), exist_ok=True)
# model.save(model_path)

# print("Model training completed and saved at:", model_path)


# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# import os

# # Define paths
# train_dir = 'data/trainsubset'
# validation_dir = '../dataset/validation'  # Optional, comment out if not using
# model_path = '../models/saved_model/deepfake_model.h5'

# # Image parameters
# IMG_HEIGHT, IMG_WIDTH = 224, 224
# BATCH_SIZE = 128  # Increased from 32
# EPOCHS = 20
# STEPS_PER_EPOCH = 500  # Custom steps to reduce from ~4000

# # Data generators with augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Validation data generator (no augmentation)
# validation_datagen = ImageDataGenerator(rescale=1./255)

# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real']
# )

# # Load validation data (optional, comment out if not using)
# validation_generator = None
# if os.path.exists(validation_dir):
#     validation_generator = validation_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(IMG_HEIGHT, IMG_WIDTH),
#         batch_size=BATCH_SIZE,
#         class_mode='binary',
#         classes=['fake', 'real']
#     )

# # Build CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Define early stopping
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) if validation_generator else None
# ]
# callbacks = [cb for cb in callbacks if cb]  # Remove None callbacks

# # Train model
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator else None,
#     callbacks=callbacks
# )

# # Save model
# os.makedirs(os.path.dirname(model_path), exist_ok=True)
# model.save(model_path)

# print("Model training completed and saved at:", model_path)











# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# import os

# # Enable memory growth to prevent OOM errors
# physical_devices = tf.config.list_physical_devices('CPU')
# try:
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
# except:
#     pass

# # Define paths
# train_dir = '../data/trainsubset'
# validation_dir = '../dataset/validation'
# model_path = '../models/saved_model/deepfake_model.h5'

# # Reduced image dimensions and batch size to prevent memory issues
# IMG_HEIGHT, IMG_WIDTH = 128, 128  # Reduced from 224x224
# BATCH_SIZE = 32  # Reduced from 128
# EPOCHS = 20

# # Data generators with augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2  # Added validation split
# )

# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real'],
#     subset='training'  # Specify training subset
# )

# # Load validation data from the same directory
# validation_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real'],
#     subset='validation'  # Specify validation subset
# )

# # Simplified CNN model with fewer parameters
# model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(256, activation='relu'),  # Reduced from 512
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Define early stopping
# callbacks = [
#     EarlyStopping(
#         monitor='val_loss',
#         patience=3,
#         restore_best_weights=True,
#         verbose=1
#     )
# ]

# # Train model with steps_per_epoch calculation
# steps_per_epoch = train_generator.samples // BATCH_SIZE
# validation_steps = validation_generator.samples // BATCH_SIZE

# # Train model
# try:
#     history = model.fit(
#         train_generator,
#         epochs=EPOCHS,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=validation_generator,
#         validation_steps=validation_steps,
#         callbacks=callbacks,
#         workers=1,  # Reduce worker threads
#         use_multiprocessing=False  # Disable multiprocessing
#     )

#     # Save model
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     model.save(model_path)
#     print("Model training completed and saved at:", model_path)
    
# except Exception as e:
#     print("Error during training:", str(e))







#Working one






# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# import os

# # Enable memory growth
# physical_devices = tf.config.list_physical_devices('CPU')
# try:
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
# except:
#     pass

# # Define paths - corrected relative paths
# train_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'trainsubset')
# model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_model', 'deepfake_model.h5')

# # Reduced parameters to prevent OOM
# IMG_HEIGHT, IMG_WIDTH = 64, 64  # Further reduced image size
# BATCH_SIZE = 16  # Further reduced batch size
# EPOCHS = 20

# # Data generator with validation split
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real'],
#     subset='training'
# )

# # Load validation data
# validation_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real'],
#     subset='validation'
# )

# # Simplified model architecture
# model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Early stopping callback
# callbacks = [
#     EarlyStopping(
#         monitor='val_loss',
#         patience=3,
#         restore_best_weights=True,
#         verbose=1
#     )
# ]

# # Train model
# try:
#     history = model.fit(
#         train_generator,
#         epochs=EPOCHS,
#         validation_data=validation_generator,
#         callbacks=callbacks
#     )

#     # Save model
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     model.save(model_path)
#     print("Model training completed and saved at:", model_path)
    
# except Exception as e:
#     print("Error during training:", str(e))











# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
#                                    BatchNormalization, GlobalAveragePooling2D)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
# from tensorflow.keras.optimizers import Adam
# import datetime

# # Disable mixed precision since your hardware doesn't support it optimally
# tf.keras.mixed_precision.set_global_policy('float32')

# # Configure GPU memory growth
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("GPU acceleration enabled")
#     except RuntimeError as e:
#         print(f"GPU configuration error: {e}")

# # Define paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# train_dir = os.path.join(BASE_DIR, 'data', 'trainsubset')
# model_path = os.path.join(BASE_DIR, 'models', 'saved_model', 'deepfake_model.h5')
# log_dir = os.path.join(BASE_DIR, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# # Model parameters - optimized for your hardware
# IMG_HEIGHT, IMG_WIDTH = 128, 128
# BATCH_SIZE = 16  # Reduced batch size
# EPOCHS = 50

# # Data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# # Load data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real'],
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     classes=['fake', 'real'],
#     subset='validation'
# )

# # Optimized model architecture
# model = Sequential([
#     # Input block
#     Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
    
#     # Second block
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
    
#     # Third block
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),
    
#     # Fourth block
#     Conv2D(256, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     GlobalAveragePooling2D(),
    
#     # Dense layers
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Callbacks
# callbacks = [
#     EarlyStopping(
#         monitor='val_loss',
#         patience=8,
#         restore_best_weights=True,
#         verbose=1
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=4,
#         verbose=1
#     ),
#     TensorBoard(
#         log_dir=log_dir,
#         histogram_freq=1,
#         update_freq='epoch'
#     )
# ]

# # Train model
# try:
#     # Create necessary directories
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
    
#     # Print model summary
#     model.summary()
    
#     print("\nStarting training...")
#     history = model.fit(
#         train_generator,
#         epochs=EPOCHS,
#         validation_data=validation_generator,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # Save model
#     model.save(model_path)
#     print(f"\nModel saved to: {model_path}")
    
#     # Print final metrics
#     final_acc = history.history['val_accuracy'][-1]
#     print(f"\nFinal validation accuracy: {final_acc*100:.2f}%")
    
# except Exception as e:
#     print(f"Error during training: {str(e)}")
# finally:
#     tf.keras.backend.clear_session()


import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D, 
                                   BatchNormalization, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import datetime

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU acceleration enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(BASE_DIR, 'data', 'trainsubset')
model_path = os.path.join(BASE_DIR, 'models', 'saved_model', 'deepfake_model.h5')
log_dir = os.path.join(BASE_DIR, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224  # ResNet50V2 default input size
BATCH_SIZE = 16
EPOCHS = 10

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect',
    validation_split=0.2
)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['fake', 'real'],
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['fake', 'real'],
    subset='validation'
)

# Create ResNet50V2 base model
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create custom model with ResNet50V2 base
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

# Add custom classification layers
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

# Create final model
model = Model(inputs=inputs, outputs=outputs)

# Compile with optimized settings
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Print model summary
model.summary()

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
]

# Two-phase training
try:
    # Phase 1: Train only top layers
    print("\nPhase 1: Training top layers...")
    history1 = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune model
    print("\nPhase 2: Fine-tuning model...")
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Continue training
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Print final metrics
    final_acc = history2.history['val_accuracy'][-1]
    final_auc = history2.history['val_auc'][-1]
    print("\nFinal Metrics:")
    print(f"Validation Accuracy: {final_acc*100:.2f}%")
    print(f"Validation AUC: {final_auc:.4f}")
    
except Exception as e:
    print(f"Error during training: {str(e)}")
finally:
    tf.keras.backend.clear_session()