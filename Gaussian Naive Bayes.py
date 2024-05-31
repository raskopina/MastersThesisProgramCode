from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from DTA import x_train, y_train, x_test, y_test
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
# Создание и обучение Gaussian Naive Bayes классификатора
gnb = GaussianNB()
gnb.fit(x_train.reshape(x_train.shape[0], -1), y_train)
y_pred = gnb.predict(x_test.reshape(x_test.shape[0], -1))

roc_auc = roc_auc_score(y_test,  y_pred)

# Рассчет PR-AUC
pr_auc = average_precision_score(y_test, y_pred)

# Рассчет Log Loss
logloss = log_loss(y_test, y_pred)

# Вычисление метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# Вывод результатов
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
print(f"Test ROC-AUC: {roc_auc}")
print(f"Test PR-AUC: {pr_auc}")
print(f"Test Log Loss: {logloss}")