ENV = 'DEV'  # can be DEV or LIVE

get_home_url_dev = "http://127.0.0.1:8000"
post_modelinfer_url_dev = f"{get_home_url_dev}/model/"

get_home_url_live = ""
post_modelinfer_url_live = f"{get_home_url_live}/model/"


# ------------ File/Folder Locations ------------
core_filename = "logs/core_script_results.log"
unittest_filename = "logs/unit_test_results.log"

# this file should be inside data folder
input_csv_filename = "census.csv"

test_size = 0.30
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
