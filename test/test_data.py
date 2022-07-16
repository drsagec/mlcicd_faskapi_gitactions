"""
set of test data for api and train_model test
"""

test_get_data1 = {"get_home_message": 'Welcome'}
test_get_data2 = {"get_home_message": 'Welcomea'}


trainmodel_data1 = {"modelname": "svm",
                    "input_csv_filename": "census.csv",
                    "test_size": 0.3
                    }

trainmodel_data2 = {"modelname": "rf",
                    "input_csv_filename": "census.csv",
                    "test_size": 0.2
                    }

infersalary_data1 = {
    'age': 25,
    'workclass': 'State-gov',
    'fnlgt': 122272,
    'education': 'Assoc-voc',
    'education-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': ' Asian-Pac-Islander',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States',
    "modelname": "svm"}

infersalary_data1_expceted = False

infersalary_data2 = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 338409,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Female",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 39,
    "native_country": "United-States",
    "modelname": "rf"}

infersalary_data2_expceted = True
