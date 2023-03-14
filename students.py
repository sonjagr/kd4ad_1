import tensorflow as tf
from tensorflow.keras import layers

def student_555():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(2, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_585():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(4, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_393():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(4, (8,8), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (4,4), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_525():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(4, (8,8), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (4,4), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_789():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(4, (8,8), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(8, (4,4), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_1101():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(4, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_282():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(1, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_295():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(2, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_159():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(1, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student


def student_165():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(2, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(1, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_141():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (4, 4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (2, 2), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_233():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(2, (8,8), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(3, (4,4), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_425():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(2, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(3, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_150():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(1, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_216():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(1, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(3, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_75():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (4, 4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(4, (2, 2), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (4, 4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(2, (2, 2), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

def student_84():
    student = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28,28,1)),
        layers.Conv2D(1, (4,4), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(1, (8,8), activation='relu'),
        layers.MaxPooling2D(4),
        layers.Dense(1)
    ])
    print(student.summary())
    return student

