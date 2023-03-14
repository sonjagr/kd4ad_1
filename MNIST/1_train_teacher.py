import tensorflow as tf
import numpy as np
import sys, os

from tensorflow.keras import layers

### USER OPTIONS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
normal_digit = 7
lr = 1e-3
batch_size = 32
#####

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

##get training data
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = x_train.astype('float32') / 255.

## define teacher

input_img = tf.keras.Input(shape=(28, 28, 1))
act = tf.keras.layers.LeakyReLU(alpha=0.3)
x = layers.Conv2D(16, (3, 3), strides=2, activation=act, padding='same')(input_img)
x = layers.Conv2D(32, (3, 3), strides=2, activation=act, padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation=act, padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=2, activation=act, padding='same')(x)
x = layers.Conv2D(16, (3, 3), strides=2, activation=act, padding='same')(x)

flat = layers.Flatten()(x)
lin1 = layers.Dense(100, activation=act)(flat)
lin2 = layers.Dense(20)(lin1)
lin3 = layers.Dense(100, activation=act)(lin2)
lin4 = layers.Dense(2*2*16, activation=act)(lin3)
lin4_out = layers.Reshape((2,2,16), input_shape=(2*2*16,))(lin4)

x = layers.Conv2DTranspose(64, (3, 3),  strides=2, activation=act, padding='same')(lin4_out)
x = layers.Conv2DTranspose(64, (3, 3),strides=2,  activation=act, padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3),strides=2,  activation=act, padding='same')(x)
x = layers.Conv2D(16, (3, 3), strides=1, activation=act)(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), strides=1, activation='sigmoid', padding='same')(x)

teacher = tf.keras.Model(input_img, decoded)

criterion_teacher = tf.keras.losses.MeanSquaredError()
teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=criterion_teacher)
print(teacher.summary())
teacher_trainableParams = np.sum([np.prod(v.get_shape()) for v in teacher.trainable_weights])

## remove one digit from training data
print('All train samples:', len(x_train))
print('Keeping digit ', normal_digit)
x_train_normal = [x_train[key] for (key, label) in enumerate(y_train) if int(label) == normal_digit]
print('Remaining for training:', len(x_train_normal))

## split data into validation and training sets
l = np.shape(x_train_normal)[0]
train_len = int(0.9*l)

train_data_ae = x_train_normal[:train_len]
train_data_ae = np.reshape(train_data_ae, (train_len, 28, 28, 1))

val_data_ae = x_train_normal[train_len:]
val_data_ae = np.reshape(val_data_ae, (l-train_len, 28, 28, 1))

print('Train samples for AE:     ', np.shape(train_data_ae))
print('Validation samples for AE:', np.shape(val_data_ae))

train_ds = tf.data.Dataset.from_tensor_slices((train_data_ae, train_data_ae)).shuffle(7000, reshuffle_each_iteration=True, seed=42).batch(batch_size, drop_remainder=True)
val_ds = tf.data.Dataset.from_tensor_slices((val_data_ae, val_data_ae)).batch(batch_size)

## define some callbacks
mcc = tf.keras.callbacks.ModelCheckpoint(
    filepath='teachers_normal_CVPR/teacher_normal_%s' % normal_digit,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

csv_logger = tf.keras.callbacks.CSVLogger('teachers_normal_CVPR/teacher_normal_%s_log.csv' % normal_digit, append=True, separator=';')
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=11)

## train
teacher.fit(train_ds, epochs=500, validation_data = val_ds, callbacks=[es, mcc, csv_logger])



