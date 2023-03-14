import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
from students import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, help="size", required=False)
parser.add_argument("--digit", type=int, help="digit", required=False)

args, unknown = parser.parse_known_args()

given_size = args.size
given_digit = args.digit

given_size = 295
given_digit = 4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(x_train, y_train_orig), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

## same training data as for teacher, no digits removed
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = x_train.astype('float32') / 255.

## USER OPTIONS
normal_digit = given_digit
log = False
BATCH_SIZE = 1
lr = 1e-3
###

## load saved AE recon errors as targets
y_train_aed = np.load('AE_outputs/y_train_aed_teacher_normal_CVPR_%s.npy' % normal_digit)

## log transform of teacher logits
if log:
    f = lambda x: np.log(x)
    y_train_aed  = f(y_train_aed)

normal_indices = np.where(y_train_orig == normal_digit)[0]
anom_indices = np.where(y_train_orig != normal_digit)[0]

## 50/50 normal and anomalous digits for training student
normals = len(normal_indices)
np.random.seed(42)
train_indices = np.append(normal_indices, np.random.choice(anom_indices, size=normals, replace = False))

x_train = x_train[train_indices]
y_train_aed = y_train_aed[train_indices]

if given_size == 425:
    student = student_425()
elif given_size == 295:
    student = student_295()
elif given_size == 216:
    student = student_216()
elif given_size == 555:
    student = student_555()
elif given_size == 150:
    student = student_150()
elif given_size == 84:
    student = student_84()

student.summary()
student_trainableParams = np.sum([np.prod(v.get_shape()) for v in student.trainable_weights])
size = student_trainableParams
print('Number of free parameters: ', student_trainableParams)

y_train = y_train_aed
train_ds_orig = tf.data.Dataset.from_tensor_slices((x_train, y_train))

## split into training and validation sets
l = int(len(x_train)*0.9)
train_ds = train_ds_orig.take(l)
validation_ds = train_ds_orig.skip(l)
print(f"Number of training examples: {len(list(train_ds))}.")
print(f"Number of validation examples: {len(list(validation_ds))}.")

train_ds = train_ds.cache().shuffle(15000, reshuffle_each_iteration=True).batch(BATCH_SIZE)
validation_ds = validation_ds.cache().batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
student.compile(optimizer = optimizer, loss = tf.keras.losses.MeanAbsoluteError())

savename = 'students_CVPR/student_CVPR_%s_%s' % (size,normal_digit)
if log:
    savename = 'students_CVPR/student_CVPR_log_%s_%s' % (size, normal_digit)
mcc = tf.keras.callbacks.ModelCheckpoint(
    filepath=savename,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

csv_logger = tf.keras.callbacks.CSVLogger('%s_log' % (savename), append=True, separator=';')
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

print('ANOMALOUS:', normal_digit)
history = student.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=50,
    callbacks = [es, csv_logger, mcc]
)