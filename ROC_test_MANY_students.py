import sklearn.metrics
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from keras_flops import get_flops
from sklearn import preprocessing
#plt.style.use('basic_mpl_style.mplstyle')

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

## USER OPTIONS
digit = 8
dataset = 'M'
####

if dataset == 'M':
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    student_names = ['150','216', '295', '425', '555','141', 'log_141']
    base_dir = 'MNIST'
elif dataset == 'F':
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    student_names = ['216', 'log_84', 'log_75', 'log_65', 'log_55']
    base_dir = 'FASHION_MNIST'

x_test = x_test.astype('float32') / 255.

normal_indices = np.where(y_test == digit)[0]
anom_indices = np.where(y_test != digit)[0]

## 50/50 normal and anomalous digits for testing
normals = len(normal_indices)
np.random.seed(42)
test_indices = np.append(normal_indices, np.random.choice(anom_indices, size=normals, replace = False))
x_test = x_test[test_indices]
y_test = y_test[test_indices]

y_test_true = []
for i in tqdm(y_test):
    if i == digit:
        y_test_true.append(0)
    else:
        y_test_true.append(1)

teacher = tf.keras.models.load_model(os.path.join(base_dir, 'teachers_normal_CVPR/teacher_normal_%s' % digit), compile=False)
teacher_params = np.sum([np.prod(v.get_shape()) for v in teacher.trainable_weights])
teacher_flops = get_flops(teacher, batch_size=1)/ 10 ** 3

fs = 12

for name in student_names:
    try:
        student = tf.keras.models.load_model(os.path.join(base_dir,'students_CVPR/student_CVPR_%s_%s' % (name,digit)), compile=False)
    except:
        continue

    student_flops = get_flops(student, batch_size=1)/ 10 ** 3

    print(teacher.summary())
    print(student.summary())

    student_params = np.sum([np.prod(v.get_shape()) for v in student.trainable_weights])
    print('Student parameters: ', student_params)

    y_test_aed = np.load(os.path.join(base_dir,'AE_outputs/y_test_aed_teacher_normal_CVPR_%s.npy' % digit))

    if 'log' in name:
        f = lambda x: np.log(x)
        y_test_aed = f(y_test_aed)

    y_test_aed = y_test_aed[test_indices]

    student_preds = []
    student_times = []
    for x in tqdm(x_test):
        time1 = time.time()
        student_prediction = student(np.reshape(x, (1,28,28,1)))
        time2 = time.time()
        s_time = time2-time1
        student_times.append(s_time)
        student_preds.append(student_prediction)

    student_preds = np.array(student_preds).flatten()

    teacher_anom, teacher_norm = [],[]
    student_anom, student_norm = [],[]
    for teacher_pred, student_pred, true in zip(y_test_aed, student_preds, y_test_true):
        if true == 1:
            teacher_anom.append(teacher_pred)
            student_anom.append(student_pred)
        else:
            teacher_norm.append(teacher_pred)
            student_norm.append(student_pred)

    compression_factor = teacher_params / student_params
    print('Compression factor:', compression_factor)
    print('Student inference time average: ', np.mean(student_times))

    teacher_fpr, teacher_tpr, _ = roc_curve(y_test_true, y_test_aed)
    student_fpr, student_tpr, _ = roc_curve(y_test_true, student_preds)

    teacher_AUC = np.round(roc_auc_score(y_test_true, y_test_aed), 3)
    student_AUC = np.round(roc_auc_score(y_test_true, student_preds), 3)

    teacher_precision, teacher_recall, _ = sklearn.metrics.precision_recall_curve(y_test_true, y_test_aed)
    student_precision, student_recall, _ = sklearn.metrics.precision_recall_curve(y_test_true,student_preds)

    teacher_AP = np.round(sklearn.metrics.average_precision_score(y_test_true,y_test_aed), 3)
    student_AP = np.round(sklearn.metrics.average_precision_score(y_test_true,student_preds), 3)
    teacher_AUC_PR = np.round(sklearn.metrics.auc(teacher_recall, teacher_precision), 3)
    student_AUC_PR = np.round(sklearn.metrics.auc(student_recall, student_precision), 3)

    plt.plot(student_fpr, student_tpr, label='Student %s (AUC = %s)' % (name, student_AUC), zorder=3)

plt.plot(teacher_fpr, teacher_tpr, label = 'Teacher (AUC = %s)' % teacher_AUC, zorder = 3)
plt.plot([0,0.2,0.5,0.7,1], [0,0.2,0.5,0.7,1], linestyle = '--', color = 'gray', zorder = 3)
plt.legend(fontsize = fs)
plt.grid(zorder = 1)
plt.title('ROC curve [normal digit: %s]' % digit, fontsize = fs)
plt.xlabel('FPR', fontsize = fs)
plt.ylabel('TPR', fontsize = fs)
#plt.savefig('ROC_normal_%s.png' % digit)
plt.show()


