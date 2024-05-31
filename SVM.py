import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from Data import train_generator,validation_generator, test_generator
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
# Получение всех образцов из генераторов
X_train, y_train = [], []

for _ in range(len(train_generator)):
    batch = next(train_generator)
    X_train.extend(batch[0])
    y_train.extend(batch[1])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = [], []
for _ in range(len(test_generator)):
    batch = next(test_generator)
    X_test.extend(batch[0])
    y_test.extend(batch[1])
X_test, y_test = np.array(X_test), np.array(y_test)

X_val, y_val = [], []
for _ in range(len(validation_generator)):
    batch = next(validation_generator)
    X_val.extend(batch[0])
    y_val.extend(batch[1])
X_val, y_val = np.array(X_val), np.array(y_val)

# Преобразование размерности для SVC и KNeighborsClassifier
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)



# Метод опорных векторов (SVM)
svm_model = SVC()
svm_model.fit(X_train_flat, y_train)
y_pred = svm_model.predict(X_test_flat)
svm_accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Предсказание только одного изображения
sample_image = X_val_flat[0]  # Пример для предсказания
start_time_single = time.time()
prediction_single = svm_model.predict([sample_image])  # Передача одного изображения
inference_time_single = time.time() - start_time_single


roc_auc = roc_auc_score(y_test,  y_pred)

# Рассчет PR-AUC
pr_auc = average_precision_score(y_test, y_pred)

# Рассчет Log Loss
logloss = log_loss(y_test, y_pred)

# Рассчет метрик

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

# Вывод результатов
print("Accuracy:", svm_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("Confusion Matrix:\n", conf_matrix)

print("Model Size:", svm_model.support_vectors_.nbytes, "bytes")  # Размер модели в байтах
print("Prediction for a single image:", prediction_single)
print("Real mean:", y_val[0])
print("Inference Time for a single image:", inference_time_single)
print(f"Test ROC-AUC: {roc_auc}")
print(f"Test PR-AUC: {pr_auc}")
print(f"Test Log Loss: {logloss}")
