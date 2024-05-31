from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import pandas as pd
from tensorflow.keras.applications import MobileNet
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
# Параметры
# Пути к вашим данным
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

# Создание генераторов данных с аугментацией для тренировочного набора
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Расчет steps_per_epoch и validation_steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = val_generator.samples // val_generator.batch_size

# Загрузка предварительно обученной модели VGG16 и добавление дополнительных слоев
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Балансировка классов
class_weights = {0: 1., 1: 2.}

#  обучение модели
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weights
)


# Оценка модели

y_pred_probs = model.predict(test_generator)
y_pred = np.round(y_pred_probs).astype(int)
y_true = test_generator.classes

# Расчет метрик
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs)
pr_auc = average_precision_score(y_true, y_pred_probs)
logloss = log_loss(y_true, y_pred_probs)

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')
print(f'PR-AUC: {pr_auc:.4f}')
print(f'Log Loss: {logloss:.4f}')

input_shape = (224, 224, 3)
# Список типов аугментации и их параметров с соответствующими значениями интенсивности
augmentations = [
    ('Rotation', 'rotation_range', [10, 20, 30, 40, 50]),
    ('Width_Shift', 'width_shift_range', [0.1, 0.2, 0.3, 0.4, 0.5]),
    ('Height_Shift', 'height_shift_range', [0.1, 0.2, 0.3, 0.4, 0.5]),
    ('Shear', 'shear_range', [0.1, 0.2, 0.3, 0.4, 0.5]),
    ('Zoom', 'zoom_range', [0.1, 0.2, 0.3, 0.4, 0.5]),
    ('Horizontal_Flip', 'horizontal_flip', [True, False]),
    ('Brightness', 'brightness_range', [(0.8, 1.2), (0.7, 1.3), (0.6, 1.4), (0.5, 1.5), (0.4, 1.6)]),
    ('Channel_Shift', 'channel_shift_range', [10, 20, 30, 40, 50])
]

# Создаем DataFrame для сохранения результатов
results = []
# Применяем каждое искажение с разной интенсивностью и вычисляем метрики
for aug_name, param_name, intensities in augmentations:
    for intensity in intensities:
        print(f"Applying {aug_name} with intensity {intensity}")
        if param_name == 'horizontal_flip':
            test_datagen_aug = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=intensity)
        else:
            test_datagen_aug = ImageDataGenerator(rescale=1.0 / 255, **{param_name: intensity})

        test_generator_aug = test_datagen_aug.flow_from_directory(
            test_dir,
            target_size=input_shape[:2],
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )

        predictions_aug = model.predict(test_generator_aug)
        predicted_classes_aug = np.round(predictions_aug).flatten()
        true_classes = test_generator_aug.classes

        accuracy = accuracy_score(true_classes, predicted_classes_aug)
        recall = recall_score(true_classes, predicted_classes_aug)
        precision = precision_score(true_classes, predicted_classes_aug)
        f1 = f1_score(true_classes, predicted_classes_aug)
        roc_auc = roc_auc_score(true_classes, predictions_aug)
        pr_auc = average_precision_score(true_classes, predictions_aug)
        logloss = log_loss(true_classes, predictions_aug)

        results.append({
            'Augmentation': aug_name,
            'Intensity': intensity,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1_score': f1,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'Log_Loss': logloss
        })

# Сохранение результатов в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('MobileNet_analysis_results.csv', index=False)
