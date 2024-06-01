import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# Функция для анализа датасета
def analyze_dataset(dataset_dir, dataset_name, name): # Функция для анализа датасета
    # Проверка наличия папок с данными
    if not os.path.exists(dataset_dir):
        print(f"Директория {dataset_name} с данными не найдена.")
        return
    # Список классов (названий папок)
    classes = os.listdir(dataset_dir)
    # Создание DataFrame для хранения информации о классах
    class_info = pd.DataFrame(columns=['Class', 'Count'])
    # Заполнение DataFrame информацией о классах
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        num_samples = len(os.listdir(class_dir))
        class_info = class_info._append({'Class': class_name, 'Count': num_samples}, ignore_index=True)
    # Вывод основной информации о классах
    print(f"Основная информация о классах в датасете {dataset_name}:")
    print(class_info)
    # Построение графика распределения здоровых и больных легких
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")  # Установка стиля сетки
    sns.barplot(x='Class', y='Count', data=class_info, palette='husl')  # Использование цветовой схемы husl
    plt.title(f'Распределение классов в {dataset_name} датасете {name}')
    plt.xlabel('Классы')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45)
    plt.show()

# Путь к директориям с тренировочным, валидационным и тестовым датасетами
train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'
# Анализ тренировочного датасета
analyze_dataset(train_dir, 'тренировочном','(train)' )
# Анализ валидационного датасета
analyze_dataset(val_dir, 'валидационном', '(val)')
# Анализ тестового датасета
analyze_dataset(test_dir, 'тестовом', '(test)')


# Функция для поиска пути к одному изображению заданного класса в директории
def find_image_path(class_name):
    class_dir = os.path.join(train_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        return image_path

# Создание подграфиков
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Загрузка и вывод изображения больных легких
pneumonia_image_path = find_image_path('PNEUMONIA')
pneumonia_image = cv2.imread(pneumonia_image_path)
pneumonia_image_rgb = cv2.cvtColor(pneumonia_image, cv2.COLOR_BGR2RGB)
axes[0].imshow(pneumonia_image_rgb)
axes[0].set_title('Пневмония')

# Загрузка и вывод изображения здоровых легких
normal_image_path = find_image_path('NORMAL')
normal_image = cv2.imread(normal_image_path)
normal_image_rgb = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
axes[1].imshow(normal_image_rgb)
axes[1].set_title('Здоровые легкие')
# Отображение графика
plt.show()
