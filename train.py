import argparse
import os
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


def load_dataset(train_dir, test_dir):
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(train_dir, file) for file in os.listdir(train_dir) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)
    
    input_files = [ os.path.join(test_dir, file) for file in os.listdir(test_dir) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    test_data = pd.concat(raw_data)
    
    return train_data, test_data
    
    
def train(train_data, test_data, hyperparameters):
    # labels are in the first column
    train_y = train_data.iloc[:,0]
    train_X = train_data.iloc[:,1:]
    test_y = test_data.iloc[:,0]
    test_X = test_data.iloc[:,1:]
    
    # We determine the number of leaf nodes using the hyper-parameter above.
    max_leaf_nodes = hyperparameters['max_leaf_nodes']

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(train_X, train_y)
    print(f'Model Accuracy: {accuracy_score(test_y, clf.predict(test_X))}')
    
    return clf
    

def save_model(clf, model_dir):
    # Save the decision tree model.
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))


# Inference script
# https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#sagemaker-scikit-learn-model-server
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == '__main__':
    print("Training Started")
    
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=-1) #/opt/ml/input/config/hyperparameters.json
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR']) #/opt/ml/output
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR']) #/opt/ml/model -> final model from /opt/ml/model will be saved to S3 at the end of training session, before the container is destroyed
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN']) #/opt/ml/input/data/train
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST']) #/opt/ml/input/data/test
    
    args = parser.parse_args()
    
    
    hyperparameters = {}
    hyperparameters['max_leaf_nodes'] = args.max_leaf_nodes
    
    
    train_data, test_data = load_dataset(args.train, args.test)
    clf = train(train_data, test_data, hyperparameters)
    save_model(clf, args.model_dir)
