from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

class DataframePreprocessing:
    
    DEFAULT_TARGET_THEMES = [0, 5, 6, 26, 33, 139, 163, 232, 313, 339, 
                             350, 406, 409, 555, 589, 597, 634, 660, 695, 
                             729, 766, 773, 793, 800, 810, 852, 895, 951, 975]

    OTHER_THEMES_VALUE = 4242

    def __init__(self, df, x_column_name='page_text_extract', y_column_name='tema', target_themes=DEFAULT_TARGET_THEMES, other_themes_value=OTHER_THEMES_VALUE):
        self.x_column_name = x_column_name
        self.y_column_name = y_column_name
        self.other_themes_value = other_themes_value
        self.target_themes = target_themes

        self._group_samples_by_process(df.copy())

        self.target_themes.sort()
        self.labels_freq = {}

        self._set_labels_frequency()
        self.processed_df = self._process_dataframe()

    def _group_samples_by_process(self, df):
        print('Grouping processes...')
        self.df = df.groupby('process_id').apply(self._aggregate)

    def _aggregate(self, series):
        reduced = {}
        series[self.x_column_name].drop_duplicates(inplace=True)
       
        reduced[self.x_column_name] = ' '.join(str(x) for x in series[self.x_column_name].values)
        temas = np.unique(series[self.y_column_name].values)
        reduced[self.y_column_name] = temas[~np.isnan(temas)]
        return pd.Series(reduced, index=[self.x_column_name, *[self.y_column_name]])

    def _remove_rare_samples(self, series, threshold=2):
        if self.labels_freq.get(tuple(series.tolist())) < threshold:
            return False
        return True 

    def _switch_other_themes_values(self, actual_label):
        """
            Replace the values of themes that are not in target themes
            to a unique value
        """

        modified_label = set()
        for theme in actual_label:
            if theme not in self.target_themes:
                modified_label.add(self.other_themes_value)
            else:
                modified_label.add(theme)
        return sorted(modified_label)

    def _normalize_labels(self, series):
        return np.asarray(self._switch_other_themes_values(series.tolist()))
    
    def _set_labels_frequency(self):
        print('Setting labels frequency...')
        for label in self.df[self.y_column_name]:
            normalized_label = tuple(self._switch_other_themes_values(label))

            if not self.labels_freq.get(normalized_label):
                self.labels_freq[normalized_label] = 1
            else:
                self.labels_freq[normalized_label] += 1

    def _get_distinct_themes(self):
        distinct_themes = set()
        for label in self.df['labels_with_others']:
            for theme in label:
                distinct_themes.add(theme)
        self.distinct_themes = list(sorted(distinct_themes))

    def _binarize_labels(self):
        print('Binarizing labels...')
        mlb = MultiLabelBinarizer()
        binarized_columns = mlb.fit_transform(self.df['labels_with_others'].to_numpy())
        
        self._get_distinct_themes()
        columns_names = {ix:binarized_columns[:,i] for i, ix in enumerate(self.distinct_themes)}

        return pd.concat([self.df.reset_index(drop=True), pd.DataFrame(columns_names)], axis=1) 


    def _process_dataframe(self):
        self.df['labels_with_others'] = self.df[self.y_column_name].apply(self._normalize_labels)
        print('Removing rare samples...')

        self.df = self.df[self.df['labels_with_others'].apply(self._remove_rare_samples)]
        return self._binarize_labels()
