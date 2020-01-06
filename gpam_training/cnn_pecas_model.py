import pickle
from utils import (GpamTokenizer,
                   cnn_pecas_model,
                   CallBack,
                   Y_transform,
                   )
from sklearn.model_selection import train_test_split


DEFAULT_VOCAB = pickle.loads(
    open("./default_vocab/vocab_112_bag.pk", "rb").read()
)
TYPE_PECAS = "Tag Mapeada"
BODY = "text"
MAX_FEATURES = len(DEFAULT_VOCAB)
SEQUENCE_LEN = 100
EMBEDDING_OUT = 100


class PecasModel:
    def __init__(self, dataframe, vocab=DEFAULT_VOCAB, classifier=None):
        self.dataframe = dataframe
        self.vocab = vocab
        if classifier is None:
            n_classes = len(self.dataframe[TYPE_PECAS].unique())
            self.classifier = cnn_pecas_model(n_classes, MAX_FEATURES,
                                              SEQUENCE_LEN, EMBEDDING_OUT)

        else:
            self.classifier = classifier

    def _split(self, x, y):
        X_train, X_test, Y_train, Y_test = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=0)

        return X_train, X_test, Y_train, Y_test

    def _tokenize(self, X_train, shape=100):
        tokenizer = GpamTokenizer(self.vocab, X_train)
        return tokenizer.tokenizer_with_vocab(shape)

    def train(self, split_df=False, batch_size=500, epochs=20):
        print("Training...")
        X_train, Y_train = (self.dataframe[BODY], self.dataframe[TYPE_PECAS])

        if split_df:
            X_train, X_test, Y_train, Y_test = self._split(X_train, Y_train)

        y_transform = Y_transform(self.dataframe[TYPE_PECAS])
        Y_train = y_transform.transform(Y_train)
        vector = self._tokenize(X_train)
        callback = CallBack()
        self.classifier.fit(vector, Y_train, batch_size=batch_size,
                            epochs=epochs, callbacks=[callback])

    def return_model(self):
        return self.classifier.to_json()
