# Simple AutoML tool for classic machine learning

This tool provide finding the best model by desired metric.

Also it is possible to train hyperparams to provide best accuracy

This tool works only with scikit-learn models

## Installation
```bash
git clone https://github.com/ZakharovDenis/AutoML
cd AutoML
pip install .
```
or
```bash
pip install git+https://github.com/ZakharovDenis/AutoML
```

## Usage
### If you want to run Titanic AutoML demo:
```python
from AutoML.demo import TitanicAutoML
from AutoML.AutoML import Metrics

loader = TitanicAutoML(train_params=False) #use train_params=True to make AutoML find best params(takes a lot of time)
model = loader.get_best_classifier(
    metric=Metrics.F1_SCORE, 
    get_model_without_score=True
)
res = model.predict(loader.get_random_test_data())
```
You can choose any other metric from `AutoML.Metrics` or, probably, from `scikitLearn`

### If you want to train your own models
```python
from AutoML.AutoML import AutoML

loader = AutoML(models=..., models_params=..., models_params_grid=...)
model_with_score = loader.get_best_classifier()
```
for example you can run Titanic example as
```python
from AutoML.AutoML import AutoML
from AutoML.demo import TitanicAutoML, DEMO_MODELS, DEMO_PARAM_GRID

loader = AutoML(models=DEMO_MODELS, models_params_grid=None)
loader.set_data_from_csv(TitanicAutoML.prepare_dataset, 'train.csv', 'test.csv') 
## 'train.csv', 'test.csv' has to be in your directory
model_with_score = loader.get_best_classifier()
```


