import pickle
import pandas as pd
import numpy as np
from dataframe_preprocessing import DataframePreprocessing 
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

class MultilabelTraining():
    
    X_COLUMN_NAME = 'page_text_extract'
    
    def __init__(self, df, x_column_name=X_COLUMN_NAME, classifier=PassiveAggressiveClassifier(random_state=42), vectorizer=HashingVectorizer(n_features=2**14)):
        self.mo_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.dp = DataframePreprocessing(df.copy())
        self.df = self.dp.processed_df
        self.x_column_name = x_column_name
        self.y_columns_names = self.dp.distinct_themes
    
    def _split(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            stratify=y, 
                                                            test_size=0.2,
                                                            random_state=42)

        return X_train, X_test, y_train, y_test

    def _vectorize(self, X_train):
        return self.vectorizer.fit_transform(X_train)
    
    def train(self, split_df=False):
        print(self.x_column_name, self.y_columns_names, self.df.columns)
        X_train, y_train = self.df[self.x_column_name], self.df[self.y_columns_names]
        if split_df:
            X_train, X_test, y_train, y_test = self._split(X_train, y_train)
        vector = self._vectorize(X_train)
        self.mo_classifier.fit(vector, y_train)

    def get_pickle(self):
        return pickle.dumps(self.mo_classifier)

