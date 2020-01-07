from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing


class GpamTokenizer:
    def __init__(self, vocab, texts):
        self.vocab = vocab
        self.texts = texts
        self.dict_vocab = {}

    def list_2_dict(self):
        id = 1
        for word in self.vocab:
            self.dict_vocab[word] = id
            id += 1

    def return_tokens(self, text):
        return text.split(" ")

    def transform_tokens(self, tokens):
        result_transform = []
        for token in tokens:
            try:
                id = self.dict_vocab[token]
                result_transform.append(id)
            except Exception:
                continue

        return result_transform

    def pad_vector(self, texts, num):
        reshape_v = []

        for each in tqdm(texts):
            if len(each) >= num:
                reshape_v.append(each[0:num])
            else:
                zeros = num - len(each)
                temp = each
                v_zeros = [0 for each in range(zeros)]
                temp.extend(v_zeros)
                reshape_v.append(temp)

        return reshape_v

    def tokenizer_with_vocab(self, num):
        self.list_2_dict()
        result_texts = []
        for i in tqdm(range(len(self.texts))):
            tokens = self.return_tokens(self.texts[i])
            result_transform = self.transform_tokens(tokens)
            result_texts.append(result_transform)

        result = self.pad_vector(result_texts, num)

        return np.matrix(result)


def binarize_pred(preds):
    new_preds = []
    for each in preds:
        num = max(each)
        new_list = np.asarray([0 for temp in range(len(each))])
        new_list[np.where(each == num)[0][0]] = 1
        new_preds.append(np.asarray(new_list))
    return new_preds


def cnn_pecas_model(n_class, max_features, sequence_len, embedding_out):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(
        input_dim=max_features,
        output_dim=embedding_out,
        input_length=sequence_len))

    model.add(keras.layers.Conv1D(kernel_size=4, filters=256, padding='same'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


class Y_transform:
    def __init__(self, y_all):
        self.y_all = y_all
        self.le = preprocessing.LabelEncoder()
        self.le.fit(set(y_all))

    def transform(self, y):
        return self.le.transform(y)


class CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') >= 0.98):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True
