from src.AutoML.demo import TitanicAutoML

loader = TitanicAutoML(train_params=False)
model = loader.get_best_classifier(get_model_without_score=True)
print(loader.X_test.head(9))
res = loader.X_test.iloc[1]
res = model.predict(loader.get_random_test_data())
print(1)