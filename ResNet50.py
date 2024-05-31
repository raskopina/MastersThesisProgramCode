import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import time
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
# Параметры
input_shape = (224, 224, 3)
num_classes = 2
batch_size = 32
epochs = 10  # Начальная тренировка
fine_tune_epochs = 10  # Тонкая настройка
initial_learning_rate = 0.001
fine_tune_learning_rate = 1e-5
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Загрузчики данных
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Модель ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Взвешивание классов
class_weights = {0: 1.0, 1: 1.0 * (1341 / 3875)}

# Обучение модели
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    class_weight=class_weights
)

# Тонкая настройка
# Размораживание верхних слоев базовой модели
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Компиляция модели с меньшей скоростью обучения
model.compile(optimizer=Adam(learning_rate=fine_tune_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели с тонкой настройкой
fine_tune_history = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    class_weight=class_weights
)

# Оценка модели
start_time = time.time()
predictions = model.predict(test_generator)
inference_time_per_image = (time.time() - start_time) / len(test_generator)

predicted_classes = np.round(predictions).flatten()
true_classes = test_generator.classes

accuracy = accuracy_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Рассчет ROC-AUC
roc_auc = roc_auc_score(true_classes, predictions)

# Рассчет PR-AUC
pr_auc = average_precision_score(true_classes, predictions)

# Рассчет Log Loss
logloss = log_loss(true_classes, predictions)
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1-score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Inference Time per image: {inference_time_per_image} seconds")
print(f"Test ROC-AUC: {roc_auc}")
print(f"Test PR-AUC: {pr_auc}")
print(f"Test Log Loss: {logloss}")
