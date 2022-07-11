

test_get_data1 = {"get_home_message": 'Welcome'}
test_get_data2 = {"get_home_message": 'Welcomea'}

# check with std params

test_post_data1 = {
    "input_csv_filename": "census.csv",
    "test_size": 0.3,
    "random_state": 11,
    "n_splits": 4,
    "shuffle": True,
    "cv": 5
}

# check with new params

test_post_data2 = {
    "input_csv_filename": "census.csv",
    "test_size": 0.2,
    "random_state": 10,
    "n_splits": 3,
    "shuffle": True,
    "cv": 5
}

# check if all params and rows/columns are good
test_post_data3 = {'input_csv_filename': 'census.csv',
                   'test_size': 0,
                   'random_state': 0,
                   'n_splits': 0,
                   'shuffle': True,
                   'cv': 0}

# check missing file
test_post_data4 = {'input_csv_filename': 'abs.csv',
                   'test_size': 0,
                   'random_state': 0,
                   'n_splits': 0,
                   'shuffle': 0,
                   'cv': 0}
