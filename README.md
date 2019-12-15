# gpam_training
Module to facilitate the integration of a sklearn training pipeline into a deploy and retraining system

## Install

```
pip install gpam_training
```

## Usage

### Multilabel training

First of all, it is needed to have in memory a dataframe from pandas.
The csv must be in the following format:

```csv
process_id,page_text_extract,tema
1,Lorem ipsum dolor sit amet,1
2,Lorem ipsum dolor sit amet,2
2,Lorem ipsum dolor sit amet,3
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
