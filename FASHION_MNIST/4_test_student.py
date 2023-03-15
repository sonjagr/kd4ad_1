import sklearn.metrics
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from keras_flops import get_flops
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_test = x_test.astype('float32') / 255.

## USER OPTIONS
digit = 6
student_name = '216'
####

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

teacher = tf.keras.models.load_model('teachers_normal_CVPR/teacher_normal_%s' % digit, compile=False)
student = tf.keras.models.load_model('students_CVPR/student_CVPR_%s_%s' % (student_name, digit), compile=False)
teacher_params = np.sum([np.prod(v.get_shape()) for v in teacher.trainable_weights])

teacher_flops = get_flops(teacher, batch_size=1)/ 10 ** 3
student_flops = get_flops(student, batch_size=1)/ 10 ** 3

print(teacher.summary())
print(student.summary())

try:
    teacher_time = np.load('teacher_normal_%s_time.npy' % digit)
except:
    teacher_time = -1

student_params = np.sum([np.prod(v.get_shape()) for v in student.trainable_weights])
size = student_params
print('Student parameters: ', student_params)

y_test_aed = np.load('AE_outputs/y_test_aed_teacher_normal_CVPR_%s.npy' % digit)
if 'log' in student_name:
    f = lambda x: np.log(x)
    y_test_aed = f(np.load('AE_outputs/y_test_aed_teacher_normal_CVPR_%s.npy' % digit))

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

fs = 18
maximum = max(max(teacher_anom), max(student_anom), max(teacher_norm), max(student_norm))
bins = np.arange(0,maximum, maximum/50)
if 'log' in student_name:
    bins = 50
fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(np.array(teacher_norm),bins = bins, alpha=0.9, density=True, color = '#92c5de', range=None,  histtype='step', linewidth=2, cumulative=False, bottom=None, align='mid',  zorder = 3, label = 'Teacher')
ax.hist(np.array(student_norm),bins = bins, alpha=1, density=True, color = '#0571b0', range=None,  histtype='step', linewidth=2,linestyle = ':',  cumulative=False, bottom=None, align='mid',  zorder = 3, label = 'Student')
plt.grid(zorder = 1)
plt.xlabel('Test loss', loc = 'right', fontsize = fs)
plt.title('Normal' , fontsize = fs)
ax.set_xlim(-0.01,0.26)
ax.set_ylim(0,45)
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.tick_params(axis="both", which = 'both', direction="in",top=True, right=True, labelsize = fs-2)
handles, labels = ax.get_legend_handles_labels()
custom_lines = [plt.Line2D([0], [0], color='#92c5de', linewidth=2),
                plt.Line2D([0], [0], color = '#0571b0',linestyle = ':', linewidth=2)]
plt.legend(custom_lines,labels,frameon=False, fontsize =  fs-2, handlelength=  1.5, borderpad= 0.5)
ax = fig.add_subplot(122)
ax.hist(np.array(teacher_anom), bins = bins, alpha=0.9, density=True, color = 'orange', range=None, histtype='step', linewidth=2, cumulative=False, bottom=None, align='mid',  zorder = 2, label='Teacher')
ax.hist(np.array(student_anom),bins = bins, alpha=1, density=True, color = '#d7191c', range=None,  histtype='step', linewidth=2,linestyle = ':',  cumulative=False, bottom=None, align='mid',  zorder = 3, label = 'Student')
plt.grid(zorder = 1, alpha = 0.8)
handles, labels = ax.get_legend_handles_labels()
custom_lines = [plt.Line2D([0], [0], color='orange', linewidth=2),
                plt.Line2D([0], [0], color = '#d7191c',linestyle = ':', linewidth=2)]
plt.legend(custom_lines,labels,frameon=False, fontsize =  fs-2, handlelength= 1.5, borderpad= 0.5)
plt.xlabel('Test loss', loc = 'right', fontsize = fs)
plt.title('Anomalous' , fontsize = fs)
ax.set_xlim(-0.01,0.26)
ax.set_ylim(0,45)
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.tick_params(axis="both", which = 'both', direction="in",top=True, right=True, labelsize = fs-2)
plt.suptitle('Fashion-MNIST Student 4, normal label: %s' % digit, fontsize = fs)
plt.tight_layout(rect=[0, 0, 0.98, 1])
plt.savefig('FMNIST_distr.png', dpi = 600)
plt.show()

compression_factor = teacher_params / student_params
print('Compression factor:', compression_factor)
print('Student inference time average: ', np.mean(student_times))
print('Teacher inference time average: ', teacher_time)

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

fig, ax = plt.subplots()
plt.plot(teacher_fpr, teacher_tpr, label = 'Teacher (AUC = %s)' % teacher_AUC, zorder = 3)
plt.plot(student_fpr, student_tpr, linestyle = '--', label = 'Student (AUC = %s)' % student_AUC, zorder = 3)
plt.plot([0,0.2,0.5,0.7,1], [0,0.2,0.5,0.7,1], linestyle = '--', color = 'gray', zorder = 3)
plt.legend(fontsize =  fs-2, frameon=False, handlelength= 1.5, borderpad= 0.5)
plt.grid(zorder = 1, alpha = 0.8)
plt.title('Fashion-MNIST Student 4, normal label: %s' % (digit), fontsize = fs)
plt.xlabel('False Positive Rate', loc = 'right', fontsize = fs)
plt.ylabel('True Positive Rate', loc = 'top', fontsize = fs)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.tick_params(axis="both", which = 'both', direction="in",top=True, right=True, labelsize = fs-2)
#plt.savefig('ROC_normal_%s.png' % digit)
plt.tight_layout()
plt.savefig('FMNIST_ROC.png', dpi = 600)
plt.show()

fig, ax = plt.subplots()
plt.plot(teacher_recall, teacher_precision, label = 'Teacher (AUC = %s)' % (teacher_AUC_PR), zorder = 4)
plt.plot(student_recall, student_precision, linestyle = '--',label = 'Student (AUC = %s)' % ( student_AUC_PR), zorder = 4)
plt.legend(fontsize =  fs-2, frameon=False, handlelength= 1.5, borderpad= 0.5)
plt.grid(zorder = 1, alpha = 0.8)
plt.title('Fashion-MNIST Student 4, normal label: %s' % (digit), fontsize = fs)
plt.xlabel('Recall', loc = 'right', fontsize = fs)
plt.ylabel('Precision', loc = 'top', fontsize = fs)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.tick_params(axis="both", which = 'both', direction="in",top=True, right=True, labelsize = fs-2)
#plt.savefig('PRC_normal_%s.png' % digit)
plt.tight_layout()
plt.savefig('FMNIST_PRC.png', dpi = 600)
plt.show()

np.save('student_normal_%s_time' % digit, np.mean(student_times))

print(f"teacher FLOPS: {teacher_flops:.03} k")
print(f"student FLOPS: {student_flops:.03} k")

cf = teacher_flops/student_flops
print(f"FLOPS compression factor: {np.round(cf,2)}")
print(f"Free params compression factor: {np.round(compression_factor,2)}")

print('Teacher ROC-AUC:', teacher_AUC)
print('Teacher PRC-AUC:', teacher_AUC_PR)

print('Student %s ROC-AUC:' % student_name, student_AUC)
print('Student %s PRC-AUC:'% student_name, student_AUC_PR)
