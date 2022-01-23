from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from utils.logger import App_Logger
from utils.main_utils import read_params
from xgboost import XGBRegressor


class Model_Finder:
    """
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None

    """

    def __init__(self, db_name, collection_name):

        self.config = read_params()

        self.log_writter = App_Logger()

        self.db_name = db_name

        self.collection_name = collection_name

        self.clf = RandomForestClassifier()

        self.DecisionTreeReg = DecisionTreeRegressor()

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_random_forest
        Description :   get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        Output      :   The model with the best parameters
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the get_best_params_for_random_forest method of the Model_Finder class",
        )

        try:
            self.param_grid = {
                "n_estmators": self.config["model_params"]["rf_model"]["n_estimators"],
                "criterion": self.config["model_params"]["rf_model"]["criterion"],
                "max_depth": self.config["model_params"]["rf_model"]["max_depth"],
                "max_features": self.config["model_params"]["rf_model"]["max_features"],
            }

            self.grid = GridSearchCV(
                estimator=self.clf,
                param_grid=self.param_grid,
                cv=self.config["model_params"]["cv"],
                verbose=self.config["model_params"]["verbose"],
            )

            self.grid.fit(train_x, train_y)

            self.criterion = self.grid.best_params_["criterion"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.max_features = self.grid.best_params_["max_features"]

            self.n_estimators = self.grid.best_params_["n_estimators"]

            self.clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.clf.fit(train_x, train_y)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Random Forest best params: "
                + str(self.grid.best_params_)
                + ". Exited the get_best_params_for_random_forest method of the Model_Finder class",
            )

            return self.clf

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_random_forest, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Random Forest Parameter tuning  failed. \
                    Exited the get_best_params_for_random_forest method of the Model_Finder class",
            )

            raise Exception(
                f"Exception occured in Class : Model_Finder. \
                Method : get_best_params_for_random_forest, Error : {str(e)}"
            )

    def get_best_params_for_DecisionTreeRegressor(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_DecisionTreeRegressor
        Description :   get the parameters for DecisionTreeRegressor Algorithm which give the best accuracy.
                        se Hyper Parameter Tuning.
        Output      :   The model with the best parameters
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class",
        )

        try:
            self.param_grid_decisionTree = {
                "criterion": ["mse", "friedman_mse", "mae"],
                "splitter": ["best", "random"],
                "max_features": ["auto", "sqrt", "log2"],
                "max_depth": range(2, 16, 2),
                "min_samples_split": range(2, 16, 2),
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                self.DecisionTreeReg,
                self.param_grid_decisionTree,
                verbose=3,
                cv=5,
                n_jobs=-1,
            )

            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_["criterion"]

            self.splitter = self.grid.best_params_["splitter"]

            self.max_features = self.grid.best_params_["max_features"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.min_samples_split = self.grid.best_params_["min_samples_split"]

            self.decisionTreeReg = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_jobs=-1,
            )

            self.decisionTreeReg.fit(train_x, train_y)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="KNN best params: "
                + str(self.grid.best_params_)
                + ". Exited the KNN method of the Model_Finder class",
            )

            return self.decisionTreeReg

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_DecisionTreeRegressor, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="knn Parameter tuning  failed. Exited the knn method of the Model_Finder class",
            )

            raise Exception(
                f"Exception occured in Class : Model_Finder. \
                Method : get_best_params_for_DecisionTreeRegressor, Error : {str(e)}"
            )

    def get_best_params_for_xgboost(self, train_x, train_y):

        """
        Method Name :   get_best_params_for_xgboost
        Description :   get the parameters for XGBoost Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        Output      :   The model with the best parameters
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the get_best_params_for_xgboost method of the Model_Finder class",
        )

        try:
            self.param_grid_xgboost = {
                "learning_rate": self.config["model_params"]["xgb_model"][
                    "learning_rate"
                ],
                "max_depth": self.config["model_params"]["xgb_model"]["max_depth"],
                "n_estimators": self.config["model_params"]["xgb_model"][
                    "n_estimators"
                ],
            }

            self.grid = GridSearchCV(
                XGBRegressor(objective="reg:linear"),
                self.param_grid_xgboost,
                verbose=self.config["model_params"]["verbose"],
                cv=self.config["model_params"]["cv"],
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.grid.fit(train_x, train_y)

            self.learning_rate = self.grid.best_params_["learning_rate"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.n_estimators = self.grid.best_params_["n_estimators"]

            self.xgb = XGBRegressor(
                objective="reg:linear",
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.xgb.fit(train_x, train_y)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="XGBoost best params: "
                + str(self.grid.best_params_)
                + ". Exited the get_best_params_for_xgboost method of the Model_Finder class",
            )

            return self.xgb

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_xgboost, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="XGBoost Parameter tuning  failed. \
                    Exited the get_best_params_for_xgboost method of the Model_Finder class",
            )

            raise Exception(
                f"Exception occured in Class : Model_Finder. \
                Method : get_best_params_for_xgboost, Error : {str(e)}"
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_best_model
        Description :   Find out the Model which has the best AUC score.
        Output      :   The best model name and the model object
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the get_best_model method of the Model_Finder class",
        )

        try:
            ## Decision Tree Reg
            self.decisionTreeReg = self.get_best_params_for_DecisionTreeRegressor(
                train_x, train_y
            )

            self.pred_dt_reg = self.decisionTreeReg.predict(test_x)

            self.decisionTreeReg_error = r2_score(test_y, self.pred_dt_reg)

            ## XGBoost Reg
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)

            self.pred_xgb = self.xgboost.predict(test_x)

            self.pred_xgb_error = r2_score(test_y, self.pred_xgb)

            ## Random Forest
            self.rf_reg = self.get_best_params_for_random_forest(train_x, train_y)

            self.pred_rf_reg = self.rf_reg.predict(test_x)

            self.prediction_rf_error = r2_score(test_x, self.pred_rf_reg)

            self.pred_rf_error = r2_score(test_x, self.pred_rf_reg)

            return (
                self.decisionTreeReg_error,
                self.decisionTreeReg,
                self.pred_xgb_error,
                self.pred_xgb_error,
                self.pred_rf_error,
                self.rf_reg,
            )

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Model_Finder, Method : get_trained_models, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Model Selection Failed. Exited the get_best_model method of the Model_Finder class",
            )

            raise Exception(
                f"Exception occured in Class : Model_Finder, Method : get_trained_models, Error : {str(e)}"
            )
