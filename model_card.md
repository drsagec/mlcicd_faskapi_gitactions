# Model Card

It describes the approaches used while creating the ML model, evaluating and testing before deployment.


## Model Details
### Census Income Classifier

This is a MLOps approach to continually train ML models using various approaches, finding the best model, testing it and continually deploying it to fastapi (Swagger UI) via Heroku. 
It predicts the individual income if above 50k or below it based on categorical features:  
"workclass", "education",  "marital-status", "occupation", "relationship", "race", "sex", "native-country".


owners: Udacity Nano-Degree MLOps, Dr. Sagar Chhetri

contact: n/a




## Intended Use
It is intended to be used for educational purpose only; not to be used for the production ready applications. The quality and the overall performance of models provided are not validated on all aspects like fairness, robustness and accuracy of data.

## Training and Testing Data
Datasets: credit - https://archive.ics.uci.edu/ml/datasets/census+income

Training of Model only uses categorical features:  "workclass", "education",  "marital-status", "occupation", "relationship", "race", "sex", "native-country"

Train models : Logistic Regression (lr),  K Nearest Neighbour (lr), Decision Tree (dt), Random Forest (rf), Adaboost Classifier (adb), support vector classifier (svm), Gradient Boosting Classifier (gdboost), Xtrim Gradient Boosting Classifier (xgboost)

All the models are evaluated and tested before deployment.


### Evaluation Data
Data is evaluated using pandas-profile, Exploratory Data Analysis (EDA). The data is cleaned out for missing values and '?'.

The columns are transformed to match python formats, including removal of spaces in column names. 

Model evaluation used following approaches:
- classification scores (accuracy)
- roc auc scores
- cross validation scores (mean)
- classification report

Continuous testing:
- pytest

### Performance Metrics
The performance metrics created by the model evaluation is present below for reference:

{
  "classification_scores": {
    "lr": 0.8314320814320815,
    "knn": 0.8352930852930853,
    "dt": 0.868945243945244,
    "rf": 0.868945243945244,
    "adb": 0.8273955773955773,
    "svm": 0.8361705861705861,
    "gdboost": 0.8333187083187084,
    "xgboost": 0.8459108459108459
  },
  "best_classification_scores": {
    "best_model_name": "dt",
    "best_model_score": 0.868945243945244
  },
  "roc_auc_scores": {
    "lr": 0.732348264850428,
    "knn": 0.730196382949681,
    "dt": 0.7244058627826716,
    "rf": 0.7365479261217678,
    "adb": 0.7270257959859893,
    "svm": 0.7300785719265295,
    "gdboost": 0.7285518031647872,
    "xgboost": 0.7379197713376502
  },
  "best_roc_auc_scores": {
    "best_model_name": "xgboost",
    "best_model_score": 0.7379197713376502
  },
  "cross_val_scores_mean": {
    "lr": 0.8316085368480579,
    "knn": 0.809588147462399,
    "dt": 0.8155769541997087,
    "rf": 0.8234698293830031,
    "adb": 0.8284452074871236,
    "svm": 0.830809888444619,
    "gdbst": 0.830717845163953,
    "xgbst": 0.8303799837482473
  },
  "best_cross_val_scores_mean": {
    "best_model_name": "knn",
    "best_model_score": 0.809588147462399
  },
  "rf_report": {
    "0": {
      "precision": 0.873514211886305,
      "recall": 0.9026702269692924,
      "f1-score": 0.8878529218647406,
      "support": 7490
    },
    "1": {
      "precision": 0.6407097092163627,
      "recall": 0.5704256252742431,
      "f1-score": 0.6035283194057567,
      "support": 2279
    },
    "accuracy": 0.8251612242808886,
    "Macro avg": {
      "precision": 0.7571119605513339,
      "recall": 0.7365479261217678,
      "f1-score": 0.7456906206352487,
      "support": 9769
    },
    "Weighted avg": {
      "precision": 0.8192034880061946,
      "recall": 0.8251612242808886,
      "f1-score": 0.8215231266959389,
      "support": 9769
    }
  }
}

## Considerations
### limitations
The data is not normally distributed, not pure random samples. During analysis we found biases around the sex, race and even salary.
                      
                      
### Ethical Considerations

The datasets used is although public, it contains features which may not be socially and morally acceptable for all groups. Data is not inclusive for all gender types, racial and generations (age groups). 

The predictions made here are only for educational use.

                                       
                                            
## Caveats and Recommendations

- The dataset gathered must be more homogeneous (random), creating less bias.
- Better strategy should be planned for data cleaning.
- The more data will create better predictions.
- Heroku used is free version; paid version will have no issue with memory overrun.
- Training data with other types like Neural Networks may produce better result.