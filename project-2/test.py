from unittest import TextTestResult
import pandas as pd
import pickle
import pickle_compat
from train import calc_glucose_features
from sklearn import preprocessing
from sklearn import decomposition

pickle_compat.patch()

with open("model.pkl", "rb") as file:
    model = pickle.load(file)
    test_data_frame = pd.read_csv("test.csv", header=None)

glucose_features = calc_glucose_features(test_data_frame)
fit_ss = preprocessing.StandardScaler().fit_transform(glucose_features)

pca = decomposition.PCA(n_components=5)
fit_pca = pca.fit_transform(fit_ss)

pd.DataFrame(model.predict(fit_pca)).to_csv("Results.csv", header=None, index=False)
