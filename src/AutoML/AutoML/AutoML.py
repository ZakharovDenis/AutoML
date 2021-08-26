import abc
from enum import Enum
from typing import Union
from random import random
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV


class Metrics(Enum):
    ACCURACY_SCORE = (accuracy_score,)
    CLASSIFICATION_REPORT = (classification_report,)
    CONFUSION_MATRIX = (confusion_matrix,)
    F1_SCORE = (f1_score,)
    FBETA_SCORE = (fbeta_score,)
    HAMMING_LOSS = (hamming_loss,)
    HINGE_LOSS = (hinge_loss,)
    JACCARD_SCORE = (jaccard_score,)
    LOG_LOSS = (log_loss,)
    MATTHEWS_CORRCOEF = (matthews_corrcoef,)
    PRECISION_SCORE = (precision_score,)
    RECALL_SCORE = (recall_score,)
    ZERO_ONE_LOSS = (zero_one_loss,)


class Model():
    '''
    Provides interfaces for AutoML
    '''
    def __init__(self, model, params=None) -> None:
        assert isinstance(model, (type, abc.ABCMeta)
                          ), 'Provide not inited model'
        self.model = model
        self.params = params or {}
        self.trained_model = None

    def set_params(self, params):
        self.params = params

    def find_best_params(self, param_grid, X_train, Y_train):
        model = self.model()
        clf = GridSearchCV(
            estimator=model, param_grid=param_grid, n_jobs=-1)
        clf.fit(X_train, Y_train)
        self.set_params(clf.best_params_)

    def train(self, X_train, Y_train):
        model = self.model(**self.params)
        model.fit(X_train, Y_train)
        self.trained_model = model

    def eval(self, X_test, Y_true, metric=None):
        assert self.trained_model is not None, 'Train model first'

        if metric is None:
            score = self.trained_model.score(X_test, Y_true)
        else:
            Y_pred = self.predict(X_test)
            score = metric(Y_true, Y_pred)
        return round(score * 100, 2)

    def predict(self, data):
        return self.trained_model.predict(data)


class AutoML():
    '''
    Main AutoML class
    Parameters
    ----------
    models : list | dict, required
        list or dict of models (use dict for obvious mapping between models and params)
    models_params: list | dict, default=False
        list or dict of params of models to be set 
        (use dict for obvious mapping between models and params)
    models_params_grid: list | dict, default=False
        list or dict of params to be trained
        NOTE that `models_params_grid` hs higher priority than `models_params`,
        that means if you set both, params will be trained, not used from `models_params` 
        (use dict for obvious mapping between models and params)
    '''

    @dataclass
    class Response:
        alias: Union[int, str]
        model: ...
        score: float

    def __init__(
        self,
        models: Union[list, dict],
        models_params: Union[list, dict] = None,
        models_params_grid: Union[list, dict] = None
    ) -> None:
        self.models = models if isinstance(models, dict) else {
            i: m for i, m in enumerate(models)}
        self.models = {k: Model(v) if not isinstance(
            v, Model) else v for k, v in self.models.items()}
        self.models_params = models_params if isinstance(models_params, dict) else {
            i: m for i, m in enumerate(models_params)} if models_params is not None else None
        self.models_params_grid = models_params_grid if isinstance(models_params_grid, dict) else {
            i: m for i, m in enumerate(models_params_grid)} if models_params_grid is not None else None
        self.trained = False
        self.results = []

    def set_data_from_pd(self, X_train, Y_train, X_eval, Y_eval, X_test):
        self.X_train, self.Y_train, self.X_eval, self.Y_eval, self.X_test = X_train, Y_train, X_eval, Y_eval, X_test

    def set_data_from_csv(self, prepare_func=None, *csv_paths):
        data = [pd.read_csv(path) for path in csv_paths]
        if prepare_func is None:
            self.X_train, self.Y_train, self.X_eval, self.Y_eval, self.X_test = data
        else:
            self.X_train, self.Y_train, self.X_eval, self.Y_eval, self.X_test = prepare_func(
                *data)

    def get_random_test_data(self):
        return self.X_test.iloc[int(random()*len(self.X_test))].values.reshape(1, -1)

    def __set_params(self, model):
        if len(self.models.get(model).params) == 0 and self.models_params_grid and self.models_params_grid.get(model):
            self.models[model].find_best_params(
                self.models_params_grid[model], self.X_train, self.Y_train)
        elif len(self.models.get(model).params) == 0 and self.models_params and self.models_params.get(model):
            self.models[model].set_params(self.models_params[model])

    def train(self, model):
        self.__set_params(model)
        self.models[model].train(self.X_train, self.Y_train)

    def eval(self, model, metric: Metrics = None):
        return self.models[model].eval(self.X_eval, self.Y_eval, metric)

    def train_all(self):
        for model in self.models:
            self.train(model)

    def eval_all(self, metric: Metrics):
        for i, m in self.models.items():
            self.results.append((i, m, self.eval(i, metric)))

    def get_best_classifier(self, metric: Metrics = None, get_model_without_score=False):
        self.results = []
        if not self.trained:
            self.train_all()
            self.trained = True
        if metric is not None:
            metric = metric.value[0]
        self.eval_all(metric)
        self.results.sort(key=lambda x: x[-1], reverse=True)
        return self.results[0][1] if get_model_without_score else self.Response(*self.results[0])

    def get_results(self):
        return self.results
