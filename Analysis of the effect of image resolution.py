import os
import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss,average_precision_score


num_classes = 2
batch_size = 32
epochs = 30
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Подготовка для сохранения результатов
results = []

# Получение текущего рабочего каталога
current_directory = os.getcwd()
# Параметры
input_shapes = [(128, 128, 3), (224, 224, 3), (299, 299, 3)]  # Разрешения для анализа
# Цикл по разрешениям
for input_shape in input_shapes:
    print(f"Training with image resolution {input_shape[:2]}")

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

    # Модель InceptionV3
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Взвешивание классов


    # Обучение модели
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,

    )
    training_time = time.time() - start_time

    # Сохранение обученной модели

    model_save_path = os.path.join(current_directory, f'Analyz_razr_{input_shape[0]}_{input_shape[1]}.keras')
    save_model(model, model_save_path)
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
    roc_auc = roc_auc_score(true_classes, predictions)
    pr_auc = average_precision_score(true_classes, predictions)
    logloss = log_loss(true_classes, predictions)

    results.append({
        'resolution': input_shape[:2],
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'log_loss': logloss,
        'training_time': training_time,
        'inference_time_per_image': inference_time_per_image
    })

# Создание DataFrame и сохранение результатов
results_df = pd.DataFrame(results)
results_df.to_csv('resolution_analysis_results.csv', index=False)
