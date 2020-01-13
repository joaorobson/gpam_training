# gpam_training
Module to facilitate the integration of a sklearn training pipeline into a deploy and retraining system

## Install

```
pip install gpam_training
```

## Usage

### Multilabel training

First of all, it is needed to have a dataframe from pandas in memory.
The csv must be in the following format:

```csv
process_id,page_text_extract,tema
1,Lorem ipsum dolor sit amet,1
1,Lorem ipsum dolor sit amet,2
2,Lorem ipsum dolor sit amet,2
2,Lorem ipsum dolor sit amet,3
4,Lorem ipsum dolor sit amet,1
4,Lorem ipsum dolor sit amet,2
5,Lorem ipsum dolor sit amet,2
42,Lorem ipsum dolor sit amet,2
```

To train the model, do as shown bellow:

```python
from gpam_training import MultilabelTraining
import pandas as pd

df = pd.read_csv('example.csv')
model = MultilabelTraining(df)
model.train()
```

To dump a pickle file with the trained model, do the following:
 
```python
model_pickle = model.get_pickle()
```

### Configuration

## class MultilabelTraining

* df (default=pandas.DataFrame()): A pandas dataframe;
* x_column_name (default="page_text_extract"): The name of the text column;
* group_processes (default=True): Wheter the labels of the processes must be grouped or not. So, for a csv like the
one above, where there are one row for each label associated with a single process, this argument must
true;
* classifier (default=PassiveAgressiveClassifier(random_state=42)): The estimator to be used;
* vectorizer (default=HashingVectorizer(n_features=2 ** 14)): The vectorizer to be used;
* target_themes (default=DEFAULT_TARGET_THEMES): The values to be considered as target values. The ones
that are different from this list will be switched to the value of the attribute other_themes_values;
* other_themes_value (default=OTHER_THEMES_VALUE): The value 
* remove_processes_without_theme (default=True): If the processes labeled without theme (represented by the theme 0)
must be removed;
* is_incremental_training (default=False): Wheter the train is incremental or not;
* vocab_path (default=""): Path to a list of words representing a vocabulary. Words out of this list will be removed.
