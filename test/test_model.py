
"""
this is test file to test all the core training functions
Used for MLOps continous testing and deployment
author : Dr.Sage/Sagar Chhetri

"""
from . import test_data

from .. import train_model
from mlcicd_faskapi_gitactions import constants

import os
import joblib
import pandas as pd
import numpy as np


print("Running CORE Tests - please wait ... ")


def test_get_data_clean():
    """
    creates test data (uses 10 rows of tail data from original data sets)
    cleans and saves as test csv

    input :
        None, reads orignal file
    output:
        assertion on file exists
        saves as test cleaned_<org_filename>.csv

    """

    org_filename = constants.input_csv_filename
    testfile = f"test_{org_filename}"

    # ----- test data cleaning  -----

    filepath = f"../data/{org_filename}"
    org_df, msg = train_model.get_data_clean(filename=filepath)
    assert isinstance(org_df, pd.DataFrame)

    test_df = org_df.tail(10)
    assert isinstance(test_df, pd.DataFrame)

    # -----  test test_df saved  -----

    test_df.to_csv(testfile)
    assert os.path.exists(testfile)


def test_train_model_all():
    """
    test complete training train_model.py
    uses 10 rows of tail data from original data sets

    input :
        None, reads test data df, gets splitted, train, test, splice on scores
    output:
        assertion on datatypes, file locations, score limits

    """

    org_filename = constants.input_csv_filename
    testfile = f"test_{org_filename}"
    test_modelname = constants.test_modelname

    test_df = pd.read_csv(testfile)
    model, model_pth, mess = train_model.get_model_name_path(
        modelname=test_modelname)
    models_dir = model_pth.split("/")[0]
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # -----  test get train and test split  -----

    train, test = train_model.train_test_split(test_df)
    assert isinstance(train, pd.DataFrame)   # is train df
    assert isinstance(test, pd.DataFrame)   # is test df

    # ----- test get train model -----

    X_train, y_train, encoder, lb_train = train_model.process_data(
        train,
        categorical_features=constants.cat_features,
        label="salary",
        training=True)
    # is x train nd array
    assert isinstance(X_train, np.ndarray)

    # is y  train nd array
    assert isinstance(y_train, np.ndarray)

    # -----  test saving of encoder  -----

    joblib.dump(encoder, 'models/encoder.pkl')

    assert os.path.exists('models/encoder.pkl')

    # ----- test model fillting and saving  -----

    # fitting model
    model.fit(X_train, y_train)

    # saving  model
    joblib.dump(model, model_pth)
    assert os.path.exists(model_pth)

    # -----  test y train preds -----

    y_train_preds = train_model.get_inference(model, X_train)

    # is y train preds nd array
    assert isinstance(y_train_preds, np.ndarray)

    # ----- test of scores -----

    precision_scor, recall_scor, fbeta_scor = train_model.get_scores(
        y_train, y_train_preds)
    assert isinstance(precision_scor, float)
    assert 0 <= precision_scor <= 1

    assert isinstance(recall_scor, float)
    assert 0 <= recall_scor <= 1

    assert isinstance(fbeta_scor, float)
    assert 0 <= recall_scor <= 1

    # -----  test of splice -----
    X_test, y_test, _, _ = train_model.process_data(
        test,
        categorical_features=constants.cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb_train
    )

    y_test_preds = train_model.get_inference(model, X_test)

    df_sliced_metrics, _ = train_model.get_sliceson_report(df=test_df.tail(
        y_test.shape[0]), slice_on="education", y=y_test, preds=y_test_preds)

    # is sliced_metrics df
    assert isinstance(df_sliced_metrics, pd.DataFrame)


def test_isencoder_exists():
    """
    does encoder exists

    input :
        None
    output:
        assertion on encoder  exists at 'models/encoder.pkl'

    """
    assert os.path.exists('models/encoder.pkl')


def test_get_test_inf_salary():
    """
    test salary prediction

    input :
        None, reads model based on provided test_modelname
    output:
        assertion on returned data contains key = greater_than_50k

    """
    test_modelname = constants.test_modelname
    data = test_data.infersalary_data1
    data_df = pd.DataFrame([data])
    response = train_model.pred_income_rf(
        intake=data_df, modelname=test_modelname)

    assert "greater_than_50k" in response
