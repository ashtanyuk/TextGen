# ------------------------------------
# Генератор текста на основе LSTM-сети
# ------------------------------------
#
# https://www.geeksforgeeks.org/text-generation-using-recurrent-long-short-term-memory-network/

from __future__ import absolute_import, division, print_function, unicode_literals
 
import numpy as np
import tensorflow as tf
 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
 
from keras.optimizers import RMSprop
 
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import random
import sys

# Загружаем текстовый файл для обучения
with open('text.txt', 'r') as file:
    text = file.read()
 
# Выводим исходный текст   
print(text)

# Получаем алфавит
vocabulary = sorted(list(set(text)))
 
# Создаем словари, содержащие индекс символа и связываем их с символами
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

# Выводим алфавит 
print(vocabulary)

# Разбиваем текст на цепочки длины max_length
# Каждый временной шаг будет загружать очередную цепочку в сеть
max_length = 6
steps = 1
sentences = []
next_chars = []

# Создаем список цепочек и список символов, которые следуют за цепочками
for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])
    # print(text[i: i + max_length])
    # print(text[i + max_length])
     
# Создаем тренировочный набор
# Создаем битовые вектора для входных значений
# (Номер_цепочки-Номер_символа_в цепочке-Код_символа)
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
# Выходные данные
# (Номер_цепочки-Код_символа)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1


# Строим LSTM-сеть
model = Sequential()
model.add(LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)


# Вспомогательная функция to sample an index from a probability array
def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



# Helper function to generate text after the end of each epoch
def on_epoch_end(epoch, logs):
    return
    print()
    print('----- Generating text after Epoch: % d' % epoch)
 
    start_index = random.randint(0, len(text) - max_length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)
 
        generated = ''
        sentence = text[start_index: start_index + max_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
 
        for i in range(400):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]
 
            generated += next_char
            sentence = sentence[1:] + next_char
 
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end = on_epoch_end)

# # Сохраняем натренированную модель после каждой эпохи
# filepath = "weights.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor ='loss',
#                              verbose = 1, save_best_only = True,
#                              mode ='min')

# # Defining a helper function to reduce the learning rate each time
# # the learning plateaus
# reduce_alpha = ReduceLROnPlateau(monitor ='loss', factor = 0.2,
#                               patience = 1, min_lr = 0.001)
# callbacks = [print_callback, checkpoint, reduce_alpha]


# Обучение LSTM модели
# model.fit(X, y, batch_size = 128, epochs = 500, callbacks = callbacks)
model.fit(X, y, batch_size = 128, epochs = 1000)

# Генерация нового текста
def generate_text(length, diversity):
    # Случайное начало
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]
 
            generated += next_char
            sentence = sentence[1:] + next_char
    return generated
 
print(generate_text(500, 0.2))


