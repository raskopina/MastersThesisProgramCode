import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss, average_precision_score
import pandas as pd
import time

# Параметры
input_shape = (224, 224, 3)
num_classes = 2
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
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Гиперпараметры для исследования
learning_rates = [0.001, 0.0001, 0.01]
batch_sizes = [16, 32, 64]
epochs_options = [30]

# Функция для создания модели
def create_model(learning_rate):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),  # Оставляем фиксированное значение
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Подготовка для сохранения результатов
results = []

# Цикл по гиперпараметрам
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for epochs in epochs_options:
            print(f"Training with LR={learning_rate}, Batch Size={batch_size}, Epochs={epochs}")

            # Генераторы данных с текущим batch_size
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

            # Создание и обучение модели
            model = create_model(learning_rate)
            start_time = time.time()
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator
            )
            training_time = time.time() - start_time

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
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
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
results_df.to_csv('hyperparameter_results.csv', index=False)
