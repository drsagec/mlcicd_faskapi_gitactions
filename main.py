from fastapi import FastAPI
from fastapi import UploadFile
from pydantic import BaseModel, Field
from typing import Optional
import constants
import secrets
import pandas as pd
from io import BytesIO
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


app = FastAPI(title="Nano MLOps FastAPI/GitActions",
              description="Train new model, get predictions of salary over \
                50k and accuracy scores. <br><br>Use below creds:\
                    <br>username: 'nanouser'<br>password: 'nanopass'\
                        <br> uses continous test and deploy Heroku\
                            <br><br> v1.0.1",
              openapi_url="/openapi.json")

security = HTTPBasic()


def get_current_username(
        credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "nanouser")
    correct_password = secrets.compare_digest(credentials.password, "nanopass")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/users/me")
def read_current_user(username: str = Depends(get_current_username)):
    return {"username": username}


@app.get("/")
def read_root():
    """Check welcome message"""
    return {"message": "Welcome"}


# adding training data
@app.post("/uploaddatafile/")
def create_upload_file(
        file: UploadFile,
        username: str = Depends(get_current_username)):
    """
    Upload new data file for  training.\
        it can be used for model endpoint.
    """
    try:
        df = pd.read_csv(BytesIO(file.file.read()))
        df.to_csv('data/{file.filename}')
        return {"Success": True,
                "message": f"{file.filename} uploaded"
                }
    except Exception as excp:
        return {"Success": False,
                "message": f"{file.filename} failed.{str(excp)}"
                }

# training model


class ModelTraining(BaseModel):
    modelname: Optional[str] = 'rf'
    input_csv_filename: Optional[str] = constants.input_csv_filename
    test_size: Optional[float] = constants.test_size

    class Config:
        allow_population_by_field_name = True


@app.post("/model/")
def model_training(input_vals: ModelTraining,
                   username: str = Depends(get_current_username)):
    """
    Train new model and get scores.
    Provided file name and params will be used.

    model name can be \
        'rf','lr','knn','dt','adb','svm','gdboost','xgboost'

    *** If file is not uploaded then please upload the file \
        using uploaddatafile endpoint first
    *** Please note - this heroku is free and it may run out \
        of free memory. You may see 'Application Error'\
             when out of memory
    """
    try:
        import train_model

        msg = train_model.train_model(
            modelname=input_vals.modelname,
            filename=input_vals.input_csv_filename,
            test_size=input_vals.test_size)
        return {"result": f"{str(msg)}"}

    except Exception as Excp:
        return {"error occurred ": f"{str(Excp)}"}


# define data BaseModel structure for infer pred
class Features(BaseModel):
    age: int = Field(constants.defaults['age'])
    workclass: str = Field(constants.defaults['workclass'])
    fnlgt: int = Field(constants.defaults['fnlwgt'])
    education: str = Field(constants.defaults['education'])
    education_num: int = Field(
        constants.defaults['education_num'],
        alias='education-num')
    marital_status: str = Field(
        constants.defaults['marital_status'],
        alias='marital-status')
    occupation: str = Field(constants.defaults['occupation'])
    relationship: str = Field(constants.defaults['relationship'])
    race: str = Field(constants.defaults['race'])
    sex: str = Field(constants.defaults['sex'])
    capital_gain: int = Field(
        constants.defaults['capital_gain'],
        alias='capital-gain')
    capital_loss: int = Field(
        constants.defaults['capital_loss'],
        alias='capital-loss')
    hours_per_week: int = Field(
        constants.defaults['hours_per_week'],
        alias='hours-per-week')
    native_country: str = Field(
        constants.defaults['native_country'],
        alias='native-country')
    modelname: str = Field('rf')

# pred


@app.post("/infer_income")
async def pred_income(features: Features):
    '''
    Makes the prediction if salary greater or less than 50k,
    based on input provided and modelname \n

    model name can be \
        'rf','lr','knn','dt','adb','svm','gdboost','xgboost'
    '''

    intake = pd.DataFrame(features).set_index(0).T
    modelname = intake.modelname.values[0]
    intake.drop(columns=['modelname'], inplace=True)
    import train_model
    msg = train_model.pred_income_rf(intake=intake, modelname=modelname)
    return msg


# running tests via fast api
@app.post("/test/")
def model_tests():
    try:
        import test_fastapp
        message = test_fastapp.run_tests()
        return {"run_massage": "Ran successfully",
                "test result": message}
    except Exception as Excp:
        return {"run_massage": f"Faied to run. {str(Excp)}",
                "test result": ""}
