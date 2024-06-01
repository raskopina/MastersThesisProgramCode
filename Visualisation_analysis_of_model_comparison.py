import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Данные
data = {
    'Model': [
        'SVM', 'Decision Tree', 'KNN',
        'Log Regression', 'Gaussian Naive Bayes',
        'CNN', 'ResNet50 (frozen weights)', 'ResNet50', 'VGG16 (frozen weights)',
        'VGG16', 'Xception (frozen weights)', 'Xception', 'Inception (frozen weights)',
        'Inception', 'DenseNet (frozen weights)', 'DenseNet', 'MobileNet (frozen weights)',
        'MobileNet', 'NASNetMobile (frozen weights)', 'NASNetMobile'
    ],
    'Accuracy': [0.823, 0.815, 0.639, 0.681, 0.729, 0.865, 0.817, 0.901, 0.878, 0.894, 0.875, 0.927, 0.857, 0.940, 0.902, 0.927, 0.908, 0.932, 0.860, 0.894],
    'Recall': [0.823, 0.830, 0.653, 0.672, 0.710, 0.866, 0.884, 0.894, 0.987, 0.882, 0.884, 0.941, 0.933, 0.930, 0.956, 0.9641, 0.979, 0.964, 0.953, 0.976],
    'Precision': [0.822, 0.815, 0.639, 0.681, 0.708, 0.913, 0.833, 0.943, 0.844, 0.945, 0.912, 0.943, 0.852, 0.973, 0.894, 0.923, 0.886, 0.930, 0.843, 0.869],
    'F1 Score': [0.820, 0.804, 0.532, 0.674, 0.709, 0.889, 0.858, 0.918, 0.910, 0.912, 0.898, 0.942, 0.891, 0.951, 0.924, 0.943, 0.930, 0.947, 0.895, 0.920],
    'ROC AUC': [0.798, 0.767, 0.526, 0.643, 0.708, 0.864, 0.880, 0.965, 0.952, 0.966, 0.950, 0.923, 0.924, 0.944, 0.956, 0.972, 0.970, 0.974, 0.948, 0.955],
    'PR AUC': [0.811, 0.783, 0.637, 0.702, 0.747, 0.875, 0.923, 0.976, 0.968, 0.978, 0.969, 0.924, 0.952, 0.949, 0.971, 0.982, 0.981, 0.982, 0.967, 0.971],
    'Log Loss': [2.810, 2.938, 5.748, 5.084, 9.761, 2.146, 0.425, 0.279, 0.323, 0.257, 0.280, 1.149, 0.351, 0.945, 0.293, 0.254, 0.286, 0.259, 0.336, 0.339]
}

# Создание DataFrame
df = pd.DataFrame(data)

# Установка стиля для графиков
sns.set(style="whitegrid")

# Цветовая палитра
palette = sns.color_palette("magma", len(df))

# Функция для создания столбчатых диаграмм
def create_barplot(metric):
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='Model', y=metric, palette=palette)
    plt.xticks(rotation=90)
    plt.title(f'{metric} для различных моделей')
    plt.xlabel('Модель')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# Функция для создания диаграмм рассеяния
def create_scatterplot(metric):
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df, x='Model', y=metric, hue='Model', palette=palette, s=200)  # Увеличиваем размер пузырьков
    plt.xticks(rotation=90)
    plt.title(f'{metric} для различных моделей')
    plt.xlabel('Модель')
    plt.ylabel(metric)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.show()


# Функция для создания линейных графиков
def create_lineplot(metric):
    plt.figure(figsize=(14, 8))

    # Сортировка DataFrame по оси x ('Model')
    df_sorted = df.sort_values(by='Model')

    sns.lineplot(data=df_sorted, x='Model', y=metric, marker='o', color='purple')
    plt.xticks(rotation=90)
    plt.title(f'{metric} для различных моделей')
    plt.xlabel('Модель')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()



# Функция для создания графиков "ящик с усами" (box plots)
def create_boxplot(metric):
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='Model', y=metric, palette=palette)
    plt.xticks(rotation=90)
    plt.title(f'{metric} для различных моделей')
    plt.xlabel('Модель')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# Построение графиков для каждой метрики
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC AUC', 'PR AUC', 'Log Loss']
for metric in metrics:
    create_barplot(metric)
    create_scatterplot(metric)
    create_lineplot(metric)
    create_boxplot(metric)


# Нормализация данных для цвета
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Функция для создания диаграмм рассеяния
def create_scatterplot(metric):
    plt.figure(figsize=(14, 8))
    normalized_metric = normalize(df[metric])
    scatter = sns.scatterplot(data=df, x='Model', y=metric, size=normalized_metric, hue=normalized_metric, palette=palette, sizes=(100, 1000), legend=False)
    scatter.set_xticklabels(scatter.get_xticklabels(), rotation=90)
    plt.title(f'{metric} для различных моделей')
    plt.xlabel('Модель')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# Построение диаграмм рассеяния для каждой метрики
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC AUC', 'PR AUC', 'Log Loss']
for metric in metrics:
    create_scatterplot(metric)


