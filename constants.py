ENV = 'DEV'  # can be DEV or LIVE

get_home_url_dev = "http://127.0.0.1:8000"
post_modeltrain_url_dev = f"{get_home_url_dev}/model/"
post_modelinfer_url_dev = f"{get_home_url_dev}/infer_income/"

get_home_url_live = "https://census-mlops-nano.herokuapp.com"
post_modeltrain_url_live = f"{get_home_url_live}/model/"
post_modelinfer_url_live = f"{get_home_url_live}/infer_income/"

# ------------ File/Folder Locations ------------
core_log_filename = "core_script_results.log"
unittest_filename = "unit_test_results.log"

# this file should be inside data folder
input_csv_filename = "census.csv"
test_modelname = 'rf'
test_size = 0.20
random_state = 11
n_splits = 4
shuffle = True
cv = 5

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

defaults = {
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
    "native_country": "United-States"
}

#
beta = 1
zero_division = 1
