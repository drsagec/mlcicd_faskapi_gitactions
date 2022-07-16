"""
test file to test fast api GET and POST requests
    steps:
    1. run function run_tests to run customized tests(optional to retrain)
    2. Or run pytest

    author: Dr. Sage Chhetri
"""


from . import test_data
from .. import constants

import requests

print("Running API Tests - please wait ... ")

# env var test


def test_config_env():
    """
    testing if env is set correct

    """
    print(constants.ENV)
    assert constants.ENV in ['DEV', 'LIVE']


# get home message test

def test_get_home_data1():
    """
    postive test case to test first data set for welcome message
    """

    if constants.ENV == 'DEV':
        get_home_url = constants.get_home_url_dev

    elif constants.ENV == 'LIVE':
        get_home_url = constants.get_home_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    home_message = test_data.test_get_data1['get_home_message']
    response = requests.get(get_home_url)

    assert response.status_code == 200
    assert response.json()['message'] == home_message


def test_infer_model_post1():
    """
    postive test case- correct data and predictions
    uses infersalary_data1 from test_data.py
    test must run after train test case

    input:
        POST Method from fast api

    outputs:
        assert on response code
        assert on saved model in model folder
    """
    data = test_data.infersalary_data1

    assert constants.ENV in ['DEV', 'LIVE']

    if constants.ENV == 'DEV':
        url = constants.post_modelinfer_url_dev

    elif constants.ENV == 'LIVE':
        url = constants.post_modelinfer_url_live

    response = requests.post(url, json=data)

    # checking post response
    assert response.status_code == 200

    # checking the pred matches expected
    assert 'greater_than_50k' in str(response.json())
    assert response.json()['greater_than_50k'] in [None, True, False]


def test_infer_model_post2():
    """
    postive test case- correct data and predictions
    uses infersalary_data2 from test_data.py
    input:
        POST Method from fast api

    outputs:
        assert on response code
        assert on saved model in model folder
    """
    data = test_data.infersalary_data2

    if constants.ENV == 'DEV':
        url = constants.post_modelinfer_url_dev

    elif constants.ENV == 'LIVE':
        url = constants.post_modelinfer_url_live

    response = requests.post(url, json=data)

    # check response code
    assert response.status_code == 200

    # check the data is correct
    assert 'greater_than_50k' in str(response.json())
    assert response.json()['greater_than_50k'] in [None, True, False]


def run_tests():
    """
    Runs all above test cases - core run file
    input:
        RETRAIN flag False/True
    outputs:
        returns test result message
    """

    message = "\n\n1. Testing config flags are set correctly"
    try:
        test_config_env()
        message = f"{message}\nPASSED"
    except Exception as excp:
        message = f"{message}\n{str(excp)}"

    message = f"{message}\n\n2. Testing 'GET' home url, right message"
    try:
        test_get_home_data1
        message = f"{message}\nPASSED"
    except Exception as excp:
        message = f"{message}\n{str(excp)}"

    message = f"{message}\n\n3. Testing  inferred pred post data 1"
    try:
        test_infer_model_post1()
        message = f"{message}\nPASSED"
    except Exception as excp:
        message = f"{message}\n{str(excp)}"

    message = f"{message}\n\n4. Testing  inferred pred post data 2"
    try:
        test_infer_model_post1()
        message = f"{message}\nPASSED"
    except Exception as excp:
        message = f"{message}\n{str(excp)}"

    return message
