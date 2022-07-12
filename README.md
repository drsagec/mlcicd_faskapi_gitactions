# MLOps Heroku, FastAPI

Training data, deploying to GitHub, flake8 git-action, continually deploying to Heroku with Swagger UI (fastapi). 



### GitHub folder structure
#### data folder: 
contains the original data (csv) used for the training. Users can upload data to through POST Upload File endpoint via Swagger UI- https://census-mlops-nano.herokuapp.com/docs#

#### models folder: 
contains all the trained models as. pkl format

#### results folder: 
contains the final findings for each models and best performing model. 

### GitHub main files (root)
- Procfile :  Heroku setting
- README.md :  contains project description
- constants.py :  contains constants used by python files
- main.py :  contains script for FastAPI
- requirements.txt :  contains dependent modules. Note - versions are removed
- sanitycheck.py : tests the  test_fastapp.py if all required GET and POST test cases are available
- test_data.py : contains test data used by test_data.py
- test_fastapp.py : contains test case for FastAPI GET and POST. To run `pytest test_fastapp.py`
- train_model.py : core script for model training and saving.


### Accessing Swagger UI (fastapi)
Using Swagger UI, users can add new data for training, train the model and test them.

#### Live (HEROKU)
1. Open browser and got https://census-mlops-nano.herokuapp.com/docs#
2. Select the endpoint and click on 'Try it out'
3. Then click 'Execute'
4. Results and responses will be displayed in box below 'Execute' button.

** Please note Heroku used is free version and during training it may fail due to over usage of memory. You can clone the repo and run it locally. 

#### Local
1. clone the repo
`git clone https://github.com/drsagec/mlcicd_faskapi_gitactions.git`

2. CD into folder
`cd mlcicd_faskapi_gitactions`


3. Change env flag 
    - go to constants.py and change ENV=LIVE to ENV=DEV
    - save 

4. create conda env
`conda create -n [envname] "python=3.9"

5. install python dependencies
`pip install -r requirements.txt`

6. start the app
`uvicorn main:app --reload`

7. Then go to link provide on terminal, usually 'http://127.0.0.1:8000/docs#'



### Running files manually
1. CD into folder
`cd mlcicd_faskapi_gitactions` 

2. Run sanity test 
`python sanitycheck.py`

3. Run flake8:
- individual files 
`flake8 sanitycheck.py`
`flake8 main.py`
`flake8 test_fastapp.py`
`flake8 train_model.py`
- flake8all files at once
`flake8`

3. Run pytest:
`pytest test_fastapp.py`
- or run for customized tes messges.
`python -c "from test_fastapp import run_tests;run_tests()"`
- or visit via Swagger UI : https://census-mlops-nano.herokuapp.com/docs#/default/model_tests_test__post
** Note - before running tests make sure to train the model.



### Git 

- Review git action results, 
    - visit - https://github.com/drsagec/mlcicd_faskapi_gitactions/actions
    - or refer to screenshot continuous_integration.png
- Review commits 


### Model Card 
Open model_card.md for model explanation 

### Tech Stacks
python, conda, fastapi (RESTful API), Heroku, git, GitHub, dvc, flake8, pytest, ML Models, GitHub Actions, continuous delivery, continuous testing

