'''Script to train machine learning model.
    Using multiple models for best predictor
    author Dr. Sagar Chhetri
'''
# importing all constants used in this module
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import logging
import os
from sklearn.model_selection import train_test_split

import constants
import joblib

import pandas as pd
import plotly as py
py.offline.init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')


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

core_filename = "logs/core_script_results.log"
# setting up log file
logging.basicConfig(
    filename=core_filename,
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def is_file_exists(file_path):
    ''' Checks if file path valid and logs logging message
    input:
        file_path : (str) full file path name
    output:
        success: (bool) success trure or false
        message: (str) contain detail description of success and failures
    '''

    success = True
    message = ""

    if not os.path.exists(file_path):
        success = False
        message = f"{file_path} file not found"
        logging.error(message)
    else:
        message = f"{file_path} file  found"
        logging.info(message)

    return success, message


def is_dataframe(data_df):
    '''
    Checks if dataframe and has columns and rows, logs message
    input:
    data_df : (dataframe)
    output:
    success: (bool) success trure or false
    message: (str) contain detail of success and failures
    '''
    success = True
    message = ""

    try:
        assert isinstance(data_df, pd.DataFrame)
        # Check if columns and rows exists
        try:
            assert data_df.shape[0] > 0
            assert data_df.shape[1] > 0
            message = f"{message} | rows {data_df.shape[0]}"
            message = f"{message} , columns {data_df.shape[1]}"

        except AssertionError as err:
            message = f"{message} | No rows and columns found {str(err)}"

    except AssertionError as err:
        message = f"{message} | Not a dataframe {str(err)}"

    # logging message to log file
    if success:
        logging.info(message)
    else:
        logging.error(message)

    return success, message


def import_data(pth):
    '''
    returns dataframe for the csv at pth, sucess flag and  message

    input:
        pth: a path to the csv file for  data
    output:
        df: pandas dataframe containing the data from path
        success: (bool) success trure or false
        message: (str) contain detail description of success and failures
    '''
    orig_data_df = pd.DataFrame(dtype=object)
    # cheking if path is correct, and logging sucess/fail message in logger
    success, message = is_file_exists(file_path=pth)

    if success:
        orig_data_df = pd.read_csv(rf"{pth}", index_col=False)

        # check if dataframe or not and logging sucess/fail message in logger
        success, msg = is_dataframe(data_df=orig_data_df)
        message = f"{message} |  {str(msg)}"

    return orig_data_df, success, message


def train_results(
        input_csv_filename=constants.input_csv_filename,
        test_size=0.30,
        random_state=11,
        n_splits=4,
        shuffle=True,
        cv=5):
    """
    Train the results based on data and model params
    steps:
        get data
        clean data
        split data
        encode data
        fit model
        test model
        create best performing model
        save all models
        save all results

    inputs:
        input_csv_filename: file , must be in data folder
        test_size : float <1, split of data for test/traning
        random_state: int
        n_splits: int
        shuffle: bool
        cv: int

    output:
        model results and scores , all dict objects
    """

    logging.info("***   Data for all models  *** ")

    logging.info("-------------  Getting Data ------------")
    df_census = orig_data_df = pd.DataFrame(columns=cat_features, dtype=object)
    classification_scores = "{}"
    best_classification_scores = "{}"
    roc_auc_scores = "{}"
    best_roc_auc_scores = "{}"
    cross_val_scores_mean = "{}"
    best_cross_val_scores_mean = "{}"
    rf_report = "{}"

    data_loc = f"./data/{input_csv_filename}"
    orig_data_df, success, message = import_data(data_loc)

    if success:
        logging.info("-------------  Cleanig Data ------------")
        orig_data_df.columns = [x.replace(" ", "")
                                for x in orig_data_df.columns]
        for col in orig_data_df.columns.tolist():
            orig_data_df[col] = orig_data_df[col].astype(
                str).str.replace(" ", "")

        logging.info("\tremoved spaces in rows and columns")
        logging.info(f"\tSample:\n{str(orig_data_df.head(3))}")

        # cleaning data/columns
        org_cols = cat_features.copy()
        org_cols.append("salary")
        orig_data_df = orig_data_df[org_cols]

        # created new df df_census- we may need to check orig_data_df for
        # further analysis in case of faiure
        df_census = orig_data_df.copy()

        for col in org_cols:
            df_census.drop(df_census[df_census[col] ==
                           ' ?'].index, inplace=True)
        logging.info("\tremoved spaces and  ? from data")
        df_census['salary'].value_counts().plot(kind='bar')
        df_census = df_census.replace(r"-", "", regex=True)
        logging.info("\treplaced - with null value in columns")

    else:
        logging.error(f"\tFailed: {message}")
        success = False

    if success:
        # Training and Testing all models
        logging.info(
            "***   Training and Testing all models  *** ")
        logging.info("NOTE: data will be trained for:")
        logging.info("\tLogistic Regression (lr),  KNearest Neighbour (lr)")
        logging.info("\tDeciesion Tree (dt), Random Forest (rf)")
        logging.info("\tAdaboost Classifier (adb), ")
        logging.info("\tsupport vactor classifier (svm), ")
        logging.info("\tGradient Boosting Classifier (gdboost), ")
        logging.info("\tXtrim Gredient Boosting Classifier (xgboost)")

        logging.info("CREDIT:https://www.linkedin.com/in/michaelgalarnyk/")
        logging.info("\tfor education material")

        logging.info("-------------  encoding data  ------------")

        # Converting salary and sex columns into binary
        le = LabelEncoder()
        df_census['salary'] = le.fit_transform(df_census['salary'])
        df_census['sex'] = le.fit_transform(df_census['sex'])
        df_census = pd.get_dummies(df_census, drop_first=True)
        logging.info("\tConverting salary, sex columns into binary")
        logging.info(", used LabelEncoder()")

        logging.info("-------------  Splitting data   ------------")
        try:
            X = df_census.drop(['salary'], axis=1)
            y = df_census['salary']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            lr = LogisticRegression()
            knn = KNeighborsClassifier()
            dt = DecisionTreeClassifier()
            rf = RandomForestClassifier()
            adb = AdaBoostClassifier()
            svm = SVC()
            gdboost = GradientBoostingClassifier()
            xgboost = XGBClassifier()

        except Exception as Excp:
            logging.error(f"\tFailed splitting data.{str(Excp)}")

        # Training all models
        logging.info("-------------  Training all models  ------------")
        try:
            lr.fit(X_train, y_train)
            knn.fit(X_train, y_train)
            dt.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            adb.fit(X_train, y_train)
            svm.fit(X_train, y_train)
            gdboost.fit(X_train, y_train)
            xgboost.fit(X_train, y_train)
            logging.info("\tFinished training all models")
        except Exception as Excp:
            logging.error(f"\tFailed training all models. {str(Excp)}")

        try:
            # saving models
            logging.info("-------------  Training all models  ------------")
            logging.info("\tSaving models ")
            joblib.dump(lr, "models/lr.pkl")
            logging.info(
                "\t-- saved lr model at models/lr.pkl")
            joblib.dump(knn, "models/knn.pkl")
            logging.info(
                "\t-- saved knn model at models/knn.pkl")
            joblib.dump(dt, "models/dt.pkl")
            logging.info("\t-- saved dt model at models/dt.pkl")
            joblib.dump(rf, "models/rf.pkl")
            logging.info("\t-- saved rf model at models/rf.pkl")
            joblib.dump(adb, "models/adb.pkl")
            logging.info(
                "\t-- saved adb model at models/adb.pkl")
            joblib.dump(svm, "models/svm.pkl")
            logging.info(
                "\t-- saved svm model at models/svm.pkl")
            joblib.dump(gdboost, "models/gdboost.pkl")
            logging.info(
                "\t-- saved gdboost model at models/gdboost.pkl")
            joblib.dump(xgboost, "models/xgboost.pkl")
            logging.info(
                "\t-- saved xgboost model at models/xgboost.pkl")

        except Exception as Excp:
            logging.error(f"\tFailed saving  model(s). {str(Excp)}")

        # Models classification scores
        logging.info(
            "-------------  Models classification scores  ------------")
        try:
            classification_scores = {
                "lr": lr.score(
                    X_train, y_train), "knn": knn.score(
                    X_train, y_train), "dt": dt.score(
                    X_train, y_train), "rf": rf.score(
                    X_train, y_train), "adb": adb.score(
                        X_train, y_train), "svm": svm.score(
                            X_train, y_train), "gdboost": gdboost.score(
                                X_train, y_train), "xgboost": xgboost.score(
                                    X_train, y_train)}

            best_classification_scores = {
                'best_model_name': max(
                    classification_scores,
                    key=classification_scores.get),
                'best_model_score': max(
                    classification_scores.values())}
            logging.info("\tBest model is")
            logging.info(
                f"\t{best_classification_scores['best_model_name']}")
            logging.info(
                f"\t {best_classification_scores['best_model_score']}")

            logging.info("\tAll classification scores:")

            logging.info(f"\t{classification_scores}")

            pd.DataFrame([classification_scores]).to_csv(
                "results/classification_scores.csv")
            logging.info(
                "\tsaved results/classification_scores.csv ")
            pd.DataFrame([best_classification_scores]).to_csv(
                "results/best_classification_scores.csv")
            logging.info(
                "\tsaved results/best_classification_scores.csv ")

        except Exception as Excp:
            logging.error(
                f"\tFailed creating classif scores. {str(Excp)}")

        # Models predictions
        logging.info("-------------  Models predictions  ------------")

        try:
            lr_yprad = lr.predict(X_test)
            knn_yprad = knn.predict(X_test)
            rf_yprad = rf.predict(X_test)

        except Exception as Excp:
            logging.error(
                f"\tFailed creating predictions. {str(Excp)}")

        # calculating the roc_auc_scores for each models
        logging.info(
            "-------  Calculating the roc auc scores ------")
        try:
            roc_auc_scores = {
                "lr": roc_auc_score(y_test, lr.predict(X_test)),
                'knn': roc_auc_score(y_test, knn.predict(X_test)),
                'dt': roc_auc_score(y_test, dt.predict(X_test)),
                'rf': roc_auc_score(y_test, rf.predict(X_test)),
                'adb': roc_auc_score(y_test, adb.predict(X_test)),
                'svm': roc_auc_score(y_test, svm.predict(X_test)),
                'gdboost': roc_auc_score(y_test, gdboost.predict(X_test)),
                'xgboost': roc_auc_score(y_test, xgboost.predict(X_test)),
            }

            best_roc_auc_scores = {
                'best_model_name': max(
                    roc_auc_scores,
                    key=roc_auc_scores.get),
                'best_model_score': max(
                    roc_auc_scores.values())}

            logging.info("\tBest model is")
            logging.info(
                f"\t{best_roc_auc_scores['best_model_name']}")
            logging.info(
                f"\t with {best_roc_auc_scores['best_model_score']}")

            logging.info(f"\tAll roc auc scores:  {roc_auc_scores}")

            pd.DataFrame([roc_auc_scores]).to_csv("results/roc_auc_scores.csv")
            logging.info(
                "\tsaved results at results/roc_auc_scores.csv ")
            pd.DataFrame([best_roc_auc_scores]).to_csv(
                "results/best_roc_auc_scores.csv")
            logging.info(
                "\tsaved at results/best_roc_auc_scores.csv ")

        except Exception as Excp:
            logging.error(
                f"\tFailed creating roc auc scores. {str(Excp)}")

        # printing the classification report
        logging.info(
            "-------------  RF classif report ------------")
        try:
            rf_report = classification_report(
                y_test, rf_yprad, output_dict=True)
            logging.info(
                "Random forest classif report \n {rf_report}")
            pd.DataFrame(rf_report).to_csv("results/rf_report.csv")
            logging.info(
                "\tsaved at results/rf_report.csv ")

            lr_report = classification_report(
                y_test, lr_yprad, output_dict=True)
            logging.info(
                "Random forest classif report \n {lr_report}")
            pd.DataFrame(lr_report).to_csv("results/lr_report.csv")
            logging.info(
                "\tsaved at results/lr_report.csv")
            knn_report = classification_report(
                y_test, knn_yprad, output_dict=True)
            logging.info(
                f"Random forest classif report \n {knn_report}")
            pd.DataFrame(knn_report).to_csv("results/knn_report.csv")
            logging.info(
                "\tsaved at results/knn_report.csv")
            logging.info(
                "\tWe didn't create classf report for:")
            logging.info(
                "\tdt, adb, svm, gdboost, xgboost. For future EDA")

        except Exception as Excp:
            logging.error(
                f"\tFailed creating classif report. {str(Excp)}")

        # Ramdom forest Kfold cross validation , get 3 scores as  parameter
        # -n_split -3
        logging.info(
            "-------------  Kfold Cross validation ------------")
        logging.info(f"\tn_splits={n_splits},shuffle={shuffle}, cv={cv}")

        try:
            print("Random forest Kfold Cross validation score")
            k_f = KFold(n_splits=4, shuffle=True)
            cross_val_scores_mean = {
                "lr": cross_val_score(
                    lr, X, y, cv=5).mean(), 'knn': cross_val_score(
                    knn, X, y, cv=5).mean(), 'dt': cross_val_score(
                    dt, X, y, cv=5).mean(), 'rf': cross_val_score(
                    rf, X, y, cv=5).mean(), 'adb': cross_val_score(
                        adb, X, y, cv=5).mean(), 'svm': cross_val_score(
                            svm, X, y, cv=5).mean(), 'gdbst': cross_val_score(
                                gdboost, X, y, cv=5).mean(), 'xgbst': cross_val_score(
                                    xgboost, X, y, cv=5).mean(), }

            best_cross_val_scores_mean = {
                'best_model_name': min(
                    cross_val_scores_mean,
                    key=cross_val_scores_mean.get),
                'best_model_score': min(
                    cross_val_scores_mean.values())}
            logging.info(
                f"\tBest  model is \
                    {best_cross_val_scores_mean['best_model_name']}")

            logging.info(
                f"\t\n\t with \
                    {best_cross_val_scores_mean['best_model_score']}")

            logging.info(
                f"\tAll cross val scores mean:  {cross_val_scores_mean}")

            pd.DataFrame([cross_val_scores_mean]).to_csv(
                "results/cross_val_scores_mean.csv")
            logging.info(
                "\tsaved results at \
                    results/cross_val_scores_mean.csv ")
            pd.DataFrame([best_cross_val_scores_mean]).to_csv(
                "results/best_cross_val_scores_mean.csv")
            logging.info(
                "\tsaved best results at\
                     results/best_cross_val_scores_mean.csv ")

        except Exception as Excp:
            logging.error(
                f"\tFailed creating Kfold Cross val. {str(Excp)}")

    return \
        classification_scores, best_classification_scores, \
        roc_auc_scores, best_roc_auc_scores, cross_val_scores_mean, \
        best_cross_val_scores_mean, rf_report
