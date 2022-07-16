'''
Script to train machine learning model.
    Using multiple models for best predictions on census data

    author: Dr. Sagar Chhetri

    updates: 7/15 added spliced report
            7/09 created FastAPI, Git Actions for MLOps

# used during EDA (manual)- commented out for future
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder

'''
# importing all constants used in this module
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import logging
import os
import numpy as np
import constants
import joblib
import pandas as pd
warnings.filterwarnings('ignore')


cat_features = constants.cat_features
# setting up log file
logging.basicConfig(
    filename=constants.core_log_filename,
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def process_data(
        X,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and
    a label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be
        returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

# get the scores for predication


def get_scores(y, preds):
    """
    Gets score report

    inputs:
        y :  (array) known lables - binaries
        preds : (array) predicted lables - binaries

    return:
        precision_scor, recall_scor, fbeta_scor: (flaot) score results
        msg: (str) detail message on success/failure

    """
    precision_scor = precision_score(
        y, preds, zero_division=constants.zero_division)
    recall_scor = recall_score(y, preds, zero_division=constants.zero_division)
    fbeta_scor = fbeta_score(
        y,
        preds,
        beta=constants.beta,
        zero_division=constants.zero_division)
    return precision_scor, recall_scor, fbeta_scor

# get slice report


def get_sliceson_report(df, slice_on, y, preds):
    """
    Gets sliced score report
    inputs:
        df : (dataframe) input data
        slice_on : feature on which we are splicing data
        y :  (array) known lables - binaries
        preds : (array) predicted lables - binaries
    return:
        df_sliced_metrics: (dataframe) sliced score data
        msg: (str) detail message on success/failure

    """

    df_sliced_metrics = pd.DataFrame(dtype=object)
    data = "{}"

    if slice_on not in constants.cat_features:
        msg = "Can not splice on unknown feature"
        return df_sliced_metrics, msg

    try:
        for category in df[slice_on].unique():
            filters_val = (df[slice_on] == category).values
            sample_size = len(df.loc[df[slice_on] == category]) / len(y)
            precision_scor, recall_scor, fbeta_scor = get_scores(
                y[filters_val], preds[filters_val])

            data = {"slice_on": slice_on,
                    "category": category,
                    "sample_size": sample_size,
                    "precision_score": precision_scor,
                    "recall_score": recall_scor,
                    "fbeta_score": fbeta_scor
                    }

            df_sliced_metrics = df_sliced_metrics.append(pd.DataFrame([data]))

        df_sliced_metrics = df_sliced_metrics.set_index(
            ['slice_on', 'category'])
        msg = "Success getting spliced scores"

    except Exception as excp:
        msg = f"Exception occured while getting spliced scores. {str(excp)}"

    # logging
    logging.info(f"{msg}")

    return df_sliced_metrics, msg


# get inference
def get_inference(model, X):
    """ Get model predictions
    inputs
        model : (pkl) Trained machine learning model
        X : array  prediction data
    returns
        preds : (array) predictions

    """
    predict = model.predict(X)
    return predict

# cleaning data


def get_data_clean(filename="census.csv"):
    '''
    cleans the data and saves as <filename>_cleaned.csv
    - removes spaces from column and rows

    input:
        filename: (str) a path to the csv file for  data
    output:
        df: pandas dataframe containing the cleaned data
        message: (str) contain detail description of success and failures
    '''
    if "/" in filename:
        filepath = filename
        flname = filepath.split("/")[-1]
        cleaned_filepath = filepath.replace(flname, f"cleaned_{flname}")

    else:
        filepath = f"data/{filename}"
        cleaned_filepath = filepath.replace(filename, f"cleaned_{filename}")

    data = pd.DataFrame(dtype=object)

    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        data.columns = [x.replace(" ", "") for x in data.columns]
        for col in data.columns.tolist():
            data[col] = data[col].astype(str).str.replace(" ", "")

        msg = f"\tsuccess getting data, has {data.shape[0]}\
             rows and {data.shape[1]} columns"
        data.to_csv(cleaned_filepath)
        msg = f"{msg}\n\tsaved successfully at {cleaned_filepath}"

    elif not os.path.exists(filepath):
        msg = f"\tcouldn't find the file {filepath}"

    return data, msg


# getting the model path
def get_model_name_path(modelname):
    """
    get the model path for training, inference
    input:
        modelname: (str) model name \
            ['rf','lr','knn','dt','adb','svm','gdboost','xgboost']
    returns
        model: (object) model
        model_pth: (str) path where the model is saved with modelname
        msg: (str) detail of processing
    """
    default_models = [
        'rf',
        'lr',
        'knn',
        'dt',
        'adb',
        'svm',
        'gdboost',
        'xgboost']
    model_pth = ""
    model = None

    if modelname not in default_models:
        msg = f"not a correct model name , \
            value must be in {default_models}"
    else:
        if modelname == 'rf':
            model = RandomForestClassifier()
        elif modelname == 'lr':
            model = LogisticRegression()
        elif modelname == 'knn':
            model = KNeighborsClassifier()
        elif modelname == 'dt':
            model = DecisionTreeClassifier()
        elif modelname == 'adb':
            model = AdaBoostClassifier()
        elif modelname == 'svm':
            model = SVC()
        elif modelname == 'gdboost':
            model = GradientBoostingClassifier()
        elif modelname == 'xgboost':
            model = XGBClassifier()
        model_pth = f'models/{modelname}.pkl'

        msg = f"model is {model} , will be saved at {model_pth}"

    return model, model_pth, msg


def train_model(
        modelname='',
        filename=constants.input_csv_filename,
        test_size=constants.test_size):
    """
    Train the results based on data and model params
    steps:
        1. get data
        2. clean data
        3. split data
        4. encode data
        5. fit model
        6. scores and test model
        7. saves model
        8. saves encoder

    inputs:
        input_csv_filename: file , must be in data folder
        test_size : float <1, split of data for test/traning

    returns:
        msg: (str) complete message with scores and saved locations
    """
    msg = ""

    if not modelname:
        return "model name is empty"

    msg = f"{msg}\n************ getting data, cleaning ************"

    if filename == constants.input_csv_filename:
        msg = f"{msg}\n\tusing input data from data/{filename}"

    data, mess = get_data_clean(filename=filename)
    msg = f"{msg}\n{mess}"

    msg = f"{msg}\n\n************ training model ************"
    model, model_pth, mess = get_model_name_path(modelname)
    msg = f"{msg}\n{mess}"

    # splitting data test_size
    train, test = train_test_split(data, test_size=test_size)
    msg = f"{msg}\nsplitted data split_test_size = {test_size} \
    train size = {len(train)}, test size ={len(test)}"

    # training data using process_data function.
    X_train, y_train, encoder, lb_train = process_data(
        train,
        categorical_features=constants.cat_features,
        label="salary",
        training=True)
    # test data using the process_data function.
    X_test, y_test, encoder_test, lb_test = process_data(
        test,
        categorical_features=constants.cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb_train
    )

    msg = f"{msg}\nencoded data, used catagorical features\
          = {constants.cat_features}"
    joblib.dump(encoder, 'models/encoder.pkl')
    msg = f"{msg}\nsaved encoder at  models/encoder.pkl"

    # fitting model
    model.fit(X_train, y_train)

    # saving  model
    joblib.dump(model, model_pth)
    msg = f"{msg}\nsuccess fitting model, saved at {model_pth}"

    # getting infer preds
    y_train_preds = get_inference(model, X_train)
    y_test_preds = get_inference(model, X_test)

    # getting scores for training data
    precision_scor, recall_scor, fbeta_scor = get_scores(
        y_train, y_train_preds)
    msg = f"{msg}\nTraining Scores:\n \
    \tprecision_score = {precision_scor}\n \
    \trecall_score = {recall_scor}\n \
    \tfbeta_score = {fbeta_scor}\n "

    # getting scores for test data
    precision_scor, recall_scor, fbeta_scor = get_scores(y_test, y_test_preds)
    msg = f"{msg}\nTraining Scores:\n \
    \tprecision_score = {precision_scor}\n \
    \trecall_score = {recall_scor}\n \
    \tfbeta_score = {fbeta_scor}\n "

    # getting sliced metrics

    df_sliced_metrics, msssg = get_sliceson_report(df=data.tail(
        y_test.shape[0]), slice_on="education", y=y_test, preds=y_test_preds)

    msg = f"{msg}\nGetting sliced metrics, {msssg}"

    # saving  sliced metrics
    # securing slice_output.txt for trainer reveiw
    if os.path.exists('models/slice_output.txt'):
        df_sliced_metrics.to_csv('models/test_slice_output.txt')
    else:
        df_sliced_metrics.to_csv('models/slice_output.txt')

    msg = f"{msg}\n\tsaved at models/slice_output.txt"

    logging.info("***   Model Training  *** ")
    logging.info(f"{msg}")

    return msg

# prediting income using saved midel


def pred_income_rf(intake, modelname='rf'):
    """
    predicts the income for passed params and model

    inputs:
        intake: (dataframe) dataframe with test values
        modelname: name of model , can be \
            ['rf','lr','knn','dt','adb','svm','gdboost','xgboost']

    returns:
        greater_than_50k: (bool) prediction \
            if salary greater or less than 50k, None if failed
        message: (str) description of steps nd success/fail

    """
    msg = ""
    try:
        model_, model_pth, mess_ = get_model_name_path(modelname)

        msg = f"{msg}\n***   Infer salary   *** "
        msg = f"{msg}\n---- {modelname} ---- "
        msg = f"{msg}\nintake data:\n{intake}"

        if not os.path.exists(model_pth):
            msg = f"{msg}\nerror occurred - model doesn't exist, \
                please train the model from train_model endpoint first"
        else:
            model = joblib.load(model_pth)
            encoder = joblib.load("models/encoder.pkl")
            intake.columns = [x.replace("_", "-") for x in intake.columns]
            x_data, y_data, encoder, lb = process_data(
                intake,
                categorical_features=constants.cat_features,
                training=False,
                encoder=encoder,
            )
            pred = get_inference(model, x_data)
            if int(pred[0]) == 0:
                greater_than_50k = False
            else:
                greater_than_50k = True

            msg = f"{msg}\n-- predicted greater_than_50k = {greater_than_50k}"

    except Exception as excp:
        greater_than_50k = None
        msg = f"{msg}\nerror occurred- {str(excp)}.\
             Try retraining model again"

    # logging
    logging.info(f"{msg}")

    return {"greater_than_50k": greater_than_50k,
            "message": msg}
