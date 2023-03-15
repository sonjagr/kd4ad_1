import sklearn.metrics
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import preprocessing

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_test = x_test.astype('float32') / 255.

## USER OPTIONS
digit = 9
threshold = 0.23
####

y_test_true = []
for i in tqdm(y_test):
    if i == digit:
        y_test_true.append(0)
    else:
        y_test_true.append(1)

y_train_aed_for_scaler = np.load('AE_outputs/y_train_aed_teacher_normal_CVPR_%s.npy' % digit)
y_test_aed = np.load('AE_outputs/y_test_aed_teacher_normal_CVPR_%s.npy' % digit)

teacher = tf.keras.models.load_model('teachers_normal_CVPR/teacher_normal_%s' % digit, compile=False)
print(teacher.summary())

print(len(y_test_true), len(y_test_aed))

normalizer = preprocessing.MinMaxScaler((0,1)).fit(np.array(y_train_aed_for_scaler).reshape(-1,1))
y_test_aed = normalizer.transform(np.array(y_test_aed).reshape(-1,1))

teacher_binary = np.copy(y_test_aed)
teacher_binary[teacher_binary < threshold] = 0
teacher_binary[teacher_binary >= threshold] = 1

teacher_accuracy = sklearn.metrics.accuracy_score(y_test_true, teacher_binary)
print('TEACHER Accuracy, threshold %s:' % threshold, teacher_accuracy)

#print('Teacher inference time average: ', teacher_time)

anom, norm = [],[]
for pred,true in zip(y_test_aed, y_test_true):
    if true == 1:
        anom.append(pred)
    else:
        norm.append(pred)

print(len(anom), len(norm))
plt.hist(np.array(anom), bins = 22, alpha=0.7, density=True, color = 'red')
plt.hist(np.array(norm),bins = 22, alpha=0.7, density=True, color = 'green')
plt.show()

teacher_fpr, teacher_tpr, _ = roc_curve(y_test_true, y_test_aed)
teacher_AUC = np.round(roc_auc_score(y_test_true, y_test_aed), 3)
teacher_precision, teacher_recall, _ = sklearn.metrics.precision_recall_curve(y_test_true, y_test_aed)

teacher_AP = sklearn.metrics.average_precision_score(y_test_true,y_test_aed)
t_tn, t_fp, t_fn, t_tp = sklearn.metrics.confusion_matrix(y_test_true, teacher_binary, normalize='true').ravel()

print('Teacher: ', t_tn, t_fp, t_fn, t_tp)

print('AUC: ', teacher_AUC)
teacher_AUC_PR = np.round(sklearn.metrics.auc(teacher_recall, teacher_precision), 3)
print('PRC: ', teacher_AUC_PR)

plt.plot(teacher_fpr, teacher_tpr, label = 'Teacher, AUC = %s' % teacher_AUC, zorder = 4)
plt.plot([0,0.2,0.5,0.7,1], [0,0.2,0.5,0.7,1], linestyle = '--', color = 'gray', zorder = 4)
plt.legend()
plt.grid(zorder = 1)
plt.title('Anomalous digit: %s' % digit)
plt.show()

plt.plot(teacher_recall, teacher_precision, label = 'Teacher, AP = %s' % teacher_AP, zorder = 4)
plt.legend()
plt.grid(zorder = 1)
plt.title('Anomalous digit: %s' % digit)
plt.show()


