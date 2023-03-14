import tensorflow as tf
import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

x_test = x_test.astype('float32') / 255.
x_train = x_train.astype('float32') / 255.

anomalous_digit = 8
batch_size = 100
teacher = tf.keras.models.load_model('teachers_normal_CVPR/teacher_normal_%s' % anomalous_digit, compile=False)
print(teacher.summary())

print('Processing with teacher digit ', anomalous_digit)
y_train_aed = []
i=0
x_train_len = len(x_train)
train_batches = int(x_train_len/batch_size)
for i in range(train_batches):
    image = x_train[i*batch_size:batch_size+i*batch_size, :]
    time1 = time.time()
    aed = teacher(image.reshape(batch_size,28,28,1))
    diff = np.abs(image - aed.numpy()).reshape(batch_size, 28*28)
    error = np.mean(diff, axis = 1)
    y_train_aed = np.append(y_train_aed, error)
    i = i+1
    if i % 10 == 0:
        print(i, error.shape)

y_test_aed = []
i=0
teacher_times = []
x_test_len = len(x_test)
test_batches = int(x_test_len/batch_size)
for i in range(test_batches):
    image = x_test[i*batch_size:batch_size+i*batch_size, :]
    time1 = time.time()
    aed = teacher(image.reshape(batch_size,28,28,1))
    diff = np.abs(image - aed.numpy()).reshape(batch_size, 28*28)
    error = np.mean(diff, axis = 1)
    time2 = time.time()
    t_time = time2-time1
    teacher_times.append(t_time)
    y_test_aed = np.append(y_test_aed, error)
    i = i+1
    if i % 10 == 0:
        print(i,error.shape)

y_train_aed = np.array(y_train_aed).flatten()
y_test_aed = np.array(y_test_aed).flatten()

print(len(y_train), len(y_train_aed))
print(len(y_test), len(y_test_aed))
print('Average time: ', np.mean(teacher_times))
#np.save('AE_outputs/teacher_normal_CVPR_%s_time' % anomalous_digit, np.mean(teacher_times))
#np.save('AE_outputs/y_train_aed_teacher_normal_CVPR_%s' % anomalous_digit, y_train_aed)
#np.save('AE_outputs/y_test_aed_teacher_normal_CVPR_%s' % anomalous_digit, y_test_aed)
print('DONE')
