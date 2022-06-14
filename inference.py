import os
from sklearn.externals import joblib

# Inference script
# https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#sagemaker-scikit-learn-model-server
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf