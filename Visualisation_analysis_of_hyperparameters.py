import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Данные
data = {
    'learning_rate': [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.01, 0.01, 0.01],
    'batch_size': [16, 32, 64, 16, 32, 64, 16, 32, 64],
    'epochs': [30] * 9,
    'accuracy': [0.8814, 0.8478, 0.9279, 0.9151, 0.8654, 0.9279, 0.8606, 0.8782, 0.8189],
    'recall': [0.9897, 0.9923, 0.9308, 0.9872, 0.7897, 0.9308, 0.9667, 0.9179, 0.9897],
    'precision': [0.8465, 0.8079, 0.9528, 0.8891, 0.9935, 0.9528, 0.8359, 0.8905, 0.7798],
    'f1_score': [0.9125, 0.8907, 0.9416, 0.9356, 0.8800, 0.9416, 0.8966, 0.9040, 0.8723],
    'roc_auc': [0.9673, 0.9676, 0.9803, 0.9857, 0.9837, 0.9813, 0.9373, 0.9446, 0.9248],
    'pr_auc': [0.9795, 0.9787, 0.9874, 0.9915, 0.9905, 0.9890, 0.9576, 0.9625, 0.9502],
    'log_loss': [0.3237, 0.4155, 0.2188, 0.2711, 0.4418, 0.2271, 0.4078, 0.3375, 0.6860],
    'training_time': [7557.17, 7704.20, 8274.60, 8728.52, 8254.31, 8324.59, 9436.43, 9079.34, 8726.82],
    'inference_time_per_image': [0.0290, 0.0300, 0.0358, 0.0352, 0.0353, 0.0372, 0.0385, 0.0421, 0.0403]
}

# Создание DataFrame
df = pd.DataFrame(data)

# Сохранение в таблицу
df.to_csv('hyperparameter_analysis_results.csv', index=False)

# Печать таблицы
print(df)

# Визуализация данных
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Визуализация точности в зависимости от гиперпараметров
plt.subplot(2, 2, 1)
sns.barplot(data=df, x='learning_rate', y='accuracy', hue='batch_size')
plt.title('Accuracy vs Learning Rate and Batch Size')

# Визуализация F1-score в зависимости от гиперпараметров
plt.subplot(2, 2, 2)
sns.barplot(data=df, x='learning_rate', y='f1_score', hue='batch_size')
plt.title('F1 Score vs Learning Rate and Batch Size')

# Визуализация ROC-AUC в зависимости от гиперпараметров
plt.subplot(2, 2, 3)
sns.barplot(data=df, x='learning_rate', y='roc_auc', hue='batch_size')
plt.title('ROC-AUC vs Learning Rate and Batch Size')

# Визуализация Log Loss в зависимости от гиперпараметров
plt.subplot(2, 2, 4)
sns.barplot(data=df, x='learning_rate', y='log_loss', hue='batch_size')
plt.title('Log Loss vs Learning Rate and Batch Size')

plt.tight_layout()
plt.show()
