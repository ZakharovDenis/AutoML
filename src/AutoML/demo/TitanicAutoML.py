import re
from enum import Enum
import pkg_resources

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from ..AutoML import AutoML, Model


class Classifiers(Enum):
    SDG_CLASSIFIER = 'sdg_classifier'
    RANDOM_FOREST = 'random_forest'
    LOGREG = 'logistic_regression'
    KNN = 'knn'
    GAUSS = 'gaussian'
    PERSEPTRON = 'perseptron'
    LINEAR_SVC = 'linear_svc'
    DEC_TREE = 'decision_tree'


DEMO_MODELS = {
    Classifiers.SDG_CLASSIFIER: Model(SGDClassifier),
    Classifiers.RANDOM_FOREST: Model(RandomForestClassifier),
    Classifiers.LOGREG: Model(LogisticRegression),
    Classifiers.KNN: Model(KNeighborsClassifier),
    Classifiers.GAUSS: Model(GaussianNB),
    Classifiers.PERSEPTRON: Model(Perceptron),
    Classifiers.LINEAR_SVC: Model(LinearSVC),
    Classifiers.DEC_TREE: Model(DecisionTreeClassifier),
}

DEMO_PARAMS = {
    Classifiers.SDG_CLASSIFIER: {
        'loss': 'hinge',
        'penalty': 'l2',
        'alpha': 0.0001,
        'fit_intercept': True,
        'max_iter': 1000,
        'shuffle': True,
        'learning_rate': 'adaptive',
        'eta0': 0.0001
    },
    Classifiers.RANDOM_FOREST: {
        "n_estimators": 100,
        "criterion": "entropy",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        'max_features': "auto",
        'bootstrap': True,
        'oob_score': True,
        'class_weight': "balanced"
    },
}

DEMO_PARAM_GRID = {
    Classifiers.SDG_CLASSIFIER: {
        'loss': [
            'hinge', 'log', 'modified_huber',
            'squared_hinge', 'perceptron', 'squared_loss',
            'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
        ],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.00001*10**t for t in range(1, 5, 1)],
        'fit_intercept': [True, False],
        'max_iter': list(range(1000, 2000, 100)),
        'shuffle': [True, False],
        'eta0': [0.00001, ],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    },
    Classifiers.RANDOM_FOREST: {
        "n_estimators": list(range(100, 1001, 100)),
        "criterion": ["gini", "entropy"],
        "min_samples_split": list(range(4, 40, 4)),
        "min_samples_leaf": list(range(1, 50, 5)),
        'max_features': [None, "auto", "sqrt", "log2"],
        'bootstrap': [True, False],
        'oob_score': [True, False],
        'class_weight': ["balanced", "balanced_subsample", None]
    },
    Classifiers.LOGREG: {
        'penalty': ['l2'],
        'tol': [0.1/10**t for t in range(0, 5, 1)],
        'fit_intercept': [True, False],
        'class_weight': ['balanced', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': list(range(1000, 2000, 100)),
        'warm_start': [True, False],
    },
    Classifiers.KNN: {
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': list(range(10, 100, 10)),
        'p': [1, 2],
    },
    Classifiers.PERSEPTRON: {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.1/10**t for t in range(0, 5, 1)],
        'fit_intercept': [True, False],
        'max_iter': list(range(100, 1000, 100)),
        'tol': [0.1/10**t for t in range(0, 5, 1)],
        'shuffle': [True, False],
        'early_stopping': [True, False],
        'validation_fraction': [i/10 for i in range(1, 10, 1)],
        'n_iter_no_change': list(range(1, 10, 1)),
        'warm_start': [True, False],
    },
    Classifiers.LINEAR_SVC: {
        'penalty': ['l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'tol': [0.1/10**t for t in range(0, 5, 1)],
        'multi_class': ['ovr', 'crammer_singer'],
        'fit_intercept': [True, False],
        'max_iter': list(range(1000, 2000, 100)),
    },
    Classifiers.DEC_TREE: {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'min_samples_split': list(range(2, 10, 2)),
        'max_features': ["auto", "sqrt", "log2", None],
    }
}


class TitanicAutoML(AutoML):
    '''
    Simple class to run Titanic binary classification
    Parameters
    ----------
    train_path : str, default=None
        Path to titanic train dataset. If None library dataset used
    test_path : str, default=None
        Path to titanic test dataset. If None library dataset used
    train_params: bool, default=True
        Train models to find best params or not
    '''

    def __init__(self, train_path: str = None, test_path: str = None, train_params: bool = True) -> None:
        models = DEMO_MODELS
        params = DEMO_PARAMS
        models_params_grid = DEMO_PARAM_GRID
        models_params_grid = models_params_grid if train_params else None
        super().__init__(models, models_params=params,
                         models_params_grid=models_params_grid)
        self.set_data_from_csv(
            self.prepare_dataset,
            train_path or pkg_resources.resource_filename(
                'AutoML', 'demo/train.csv'),
            test_path or pkg_resources.resource_filename(
                'AutoML', 'demo/test.csv'),
        )

    @staticmethod
    def __transform_data(*data):
        deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
        common_value = 'S'
        titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        genders = {"male": 0, "female": 1}
        ports = {"S": 0, "C": 1, "Q": 2}
        ret_data = []
        for dataset in data:
            dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
            dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
            dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
            dataset['not_alone'] = dataset['not_alone'].astype(int)
            dataset = dataset.drop(['PassengerId'], axis=1)

            dataset['Cabin'] = dataset['Cabin'].fillna("U0")
            dataset['Deck'] = dataset['Cabin'].map(
                lambda x: re.compile("([a-zA-Z]+)").search(x).group())
            dataset['Deck'] = dataset['Deck'].map(deck)
            dataset['Deck'] = dataset['Deck'].fillna(0)
            dataset['Deck'] = dataset['Deck'].astype(int)
            dataset = dataset.drop(['Cabin'], axis=1)

            mean = dataset["Age"].mean()
            std = dataset["Age"].std()
            is_null = dataset["Age"].isnull().sum()
            rand_age = np.random.randint(mean - std, mean + std, size=is_null)
            age_slice = dataset["Age"].copy()
            age_slice[np.isnan(age_slice)] = rand_age
            dataset["Age"] = age_slice
            dataset["Age"] = dataset["Age"].astype(int)
            dataset["Age"].isnull().sum()

            dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

            dataset['Fare'] = dataset['Fare'].fillna(0)
            dataset['Fare'] = dataset['Fare'].astype(int)

            dataset['Title'] = dataset.Name.str.extract(
                ' ([A-Za-z]+)\.', expand=False)
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
            dataset['Title'] = dataset['Title'].map(titles)
            dataset['Title'] = dataset['Title'].fillna(0)
            dataset = dataset.drop(['Name'], axis=1)

            dataset['Sex'] = dataset['Sex'].map(genders)
            dataset = dataset.drop(['Ticket'], axis=1)

            dataset['Embarked'] = dataset['Embarked'].map(ports)

            dataset['Age'] = dataset['Age'].astype(int)
            dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 11) & (
                dataset['Age'] <= 18), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 18) & (
                dataset['Age'] <= 22), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 22) & (
                dataset['Age'] <= 27), 'Age'] = 3
            dataset.loc[(dataset['Age'] > 27) & (
                dataset['Age'] <= 33), 'Age'] = 4
            dataset.loc[(dataset['Age'] > 33) & (
                dataset['Age'] <= 40), 'Age'] = 5
            dataset.loc[(dataset['Age'] > 40) & (
                dataset['Age'] <= 66), 'Age'] = 6
            dataset.loc[dataset['Age'] > 66, 'Age'] = 6

            dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (
                dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (
                dataset['Fare'] <= 31), 'Fare'] = 2
            dataset.loc[(dataset['Fare'] > 31) & (
                dataset['Fare'] <= 99), 'Fare'] = 3
            dataset.loc[(dataset['Fare'] > 99) & (
                dataset['Fare'] <= 250), 'Fare'] = 4
            dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
            dataset['Fare'] = dataset['Fare'].astype(int)
            dataset['Age_Class'] = dataset['Age'] * dataset['Pclass']
            dataset['Fare_Per_Person'] = dataset['Fare'] / \
                (dataset['relatives']+1)
            dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
            ret_data.append(dataset)

        return ret_data

    @staticmethod
    def prepare_dataset(train_df, test_df):
        '''
        Titanic dataset preparation function
        '''
        data = [train_df, test_df]
        train_df, test_df = TitanicAutoML.__transform_data(*data)

        eval_df = train_df[-int(len(train_df)/6):]
        train_df = train_df[:-int(len(train_df)/6)]

        Y_train = train_df['Survived']
        X_train = train_df.drop("Survived", axis=1)

        Y_eval = eval_df['Survived']
        X_eval = eval_df.drop("Survived", axis=1)

        return X_train, Y_train, X_eval, Y_eval, test_df
