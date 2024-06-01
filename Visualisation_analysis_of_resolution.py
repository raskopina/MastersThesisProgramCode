import matplotlib.pyplot as plt
import numpy as np

# Данные
sizes = ["128 х 128", "224 х 224", "299 х 299"]
accuracy = [0.900, 0.820, 0.926]
recall = [0.858, 0.728, 0.982]
precision = [0.979, 0.979, 0.907]
f1_score = [0.915, 0.835, 0.943]
roc_auc = [0.981, 0.972, 0.979]
pr_auc = [0.988, 0.981, 0.987]
log_loss = [0.279, 0.398, 0.265]
training_time = [3016, 7599, 16760]

# Построение графиков
fig, ax1 = plt.subplots(figsize=(10, 6))

# Графики для метрик
ax1.plot(sizes, accuracy, label='Accuracy', marker='o')
ax1.plot(sizes, recall, label='Recall', marker='o')
ax1.plot(sizes, precision, label='Precision', marker='o')
ax1.plot(sizes, f1_score, label='F1 мера', marker='o')
ax1.plot(sizes, roc_auc, label='ROC AUC', marker='o')
ax1.plot(sizes, pr_auc, label='PR AUC', marker='o')
ax1.set_xlabel('Размер изображения')
ax1.set_ylabel('Значение метрики')
ax1.legend(loc='upper left')
ax1.grid(True)

# Отдельный график для Log Loss и времени обучения
fig, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(sizes, log_loss, label='Log Loss', marker='o', color='r')
ax2.set_xlabel('Размер изображения')
ax2.set_ylabel('Log Loss', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.grid(True)

ax3 = ax2.twinx()
ax3.plot(sizes, training_time, label='Время обучения, секунды', marker='o', color='b')
ax3.set_ylabel('Время обучения, секунды', color='b')
ax3.tick_params(axis='y', labelcolor='b')

fig.tight_layout()
plt.title('Зависимость метрик и времени обучения от размера изображения')
plt.show()

# Данные
sizes = ["128 х 128", "224 х 224", "299 х 299"]
metrics = {
    "Accuracy": [0.900, 0.820, 0.926],
    "Recall": [0.858, 0.728, 0.982],
    "Precision": [0.979, 0.979, 0.907],
    "F1 мера": [0.915, 0.835, 0.943],
    "ROC AUC": [0.981, 0.972, 0.979],
    "PR AUC": [0.988, 0.981, 0.987],
    "Log Loss": [0.279, 0.398, 0.265],
    "Время обучения, секунды": [3016, 7599, 16760]
}

# Цветовая палитра
colors = ['#FF7F50', '#6495ED', '#FFD700', '#ADFF2F', '#FF69B4', '#8A2BE2', '#DC143C', '#00CED1']

fig, axs = plt.subplots(4, 2, figsize=(14, 18))
axs = axs.ravel()

for i, (metric, values) in enumerate(metrics.items()):
    axs[i].bar(sizes, values, color=colors[i])
    axs[i].set_title(metric)
    axs[i].set_xlabel('Размер изображения')
    axs[i].set_ylabel(metric)
    axs[i].grid(True)

fig.tight_layout()
plt.show()
