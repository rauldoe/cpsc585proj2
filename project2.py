'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

'''Importing the EMNIST letters'''
from scipy import io as sio

batch_size = 1000
# num_classes = 10
num_classes = 26
epochs = 100

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')


# https://stackoverflow.com/questions/51125969/loading-emnist-letters-dataset/53547262#53547262
mat = sio.loadmat('emnist-letters.mat')
data = mat['dataset']

x_train = data['train'][0,0]['images'][0,0]
y_train = data['train'][0,0]['labels'][0,0]
x_test = data['test'][0,0]['images'][0,0]
y_test = data['test'][0,0]['labels'][0,0]

val_start = x_train.shape[0] - x_test.shape[0]
x_val = x_train[val_start:x_train.shape[0],:]
y_val = y_train[val_start:x_train.shape[0]]
x_train = x_train[0:val_start,:]
y_train = y_train[0:val_start]


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes, dtype='float32')

y_val = tf.keras.utils.to_categorical(y_val - 1, num_classes, dtype='float32')

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs, callbacks=[earlyStop],
                    verbose=1,
                    validation_data=(x_val, y_val)
                    )
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])