

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import constants


class ModelInfer(BaseModel):
    input_csv_filename: Optional[str] = constants.input_csv_filename
    test_size: Optional[float] = constants.test_size
    random_state: Optional[int] = constants.random_state
    n_splits: Optional[float] = constants.n_splits
    shuffle: Optional[bool] = constants.shuffle
    cv: Optional[int] = constants.cv

    class Config:
        allow_population_by_field_name = True


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome"}


@app.post("/model/")
def model_inference(input_vals: ModelInfer):

    try:
        import train_model

        classification_scores, best_classification_scores, roc_auc_scores, \
            best_roc_auc_scores, cross_val_scores_mean, \
            best_cross_val_scores_mean, rf_report = train_model.train_results(
                input_csv_filename=input_vals.input_csv_filename,
                test_size=input_vals.test_size,
                random_state=input_vals.random_state,
                n_splits=input_vals.n_splits,
                shuffle=input_vals.shuffle,
                cv=input_vals.cv)

        return {"classification_scores": classification_scores,
                "best_classification_scores": best_classification_scores,
                "roc_auc_scores": roc_auc_scores,
                "best_roc_auc_scores": best_roc_auc_scores,
                "cross_val_scores_mean": cross_val_scores_mean,
                "best_cross_val_scores_mean": best_cross_val_scores_mean,
                "rf_report": rf_report,
                "message": "Success"
                }
    except Exception as Excp:
        return {"classification_scores": "{}",
                "best_classification_scores": "{}",
                "roc_auc_scores": "{}",
                "best_roc_auc_scores": "{}",
                "cross_val_scores_mean": "{}",
                "best_cross_val_scores_mean": "{}",
                "message": f"False. {str(Excp)}"
                }


@app.post("/test/")
def model_tests(retrain: bool = False):
    try:
        import test_fastapp
        message = test_fastapp.run_tests(RETRAIN=retrain)
        return {"run_massage": "Ran successfully",
                "test result": message}
    except Exception as Excp:
        return {"run_massage": f"Faied to run. {str(Excp)}",
                "test result": ""}
