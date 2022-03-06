from sklearn.ensemble import RandomForestRegressor
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params
from xgboost import XGBRegressor


class Model_Finder:
    """
    Description :   This class shall  be used to find the model with best accuracy and AUC score.
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, table_name):
        self.table_name = table_name

        self.class_name = self.__class__.__name__

        self.config = read_params()

        self.log_writer = App_Logger()

        self.model_utils = Model_Utils()

        self.rf_model = RandomForestRegressor()

        self.xgb_model = XGBRegressor(objectective="binary:logistic")

    def get_best_model_for_random_forest(self, train_x, train_y):
        """
        Method Name :   get_best_model_for_random_forest
        Description :   get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_model_for_random_forest.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        try:
            self.rf_model_name = self.model_utils.get_model_name(
                model=self.rf_model, table_name=self.table_name
            )

            self.rf_best_params = self.model_utils.get_model_params(
                model=self.rf_model,
                model_key_name="rf_model",
                x_train=train_x,
                y_train=train_y,
                table_name=self.table_name,
            )

            self.criterion = self.rf_best_params["criterion"]

            self.max_depth = self.rf_best_params["max_depth"]

            self.max_features = self.rf_best_params["max_features"]

            self.n_estimators = self.rf_best_params["n_estimators"]

            self.log_writer.log(
                table_name=self.table_name,
                log_info=f"{self.rf_model_name} model best params are {self.rf_best_params}",
            )

            rf_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
            )

            self.log_writer.log(
                table_name=self.table_name,
                log_info=f"Initialized {self.rf_model_name} with {self.rf_best_params} as params",
            )

            rf_model.fit(train_x, train_y)

            self.log_writer.log(
                table_name=self.table_name,
                log_info=f"Created {self.rf_model_name} based on the {self.rf_best_params} as params",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return rf_model

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

    def get_best_params_for_xgboost(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_xgboost
        Description :   get the parameters for XGBoost Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_params_for_xgboost.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        try:
            self.xgb_model_name = self.model_utils.get_model_name(
                model=self.xgb_model, table_name=self.table_name
            )

            self.xgb_best_params = self.model_utils.get_model_params(
                model=self.xgb_model,
                model_key_name="xgb_model",
                x_train=train_x,
                y_train=train_y,
                table_name=self.table_name,
            )

            self.learning_rate = self.xgb_best_params["learning_rate"]

            self.max_depth = self.xgb_best_params["max_depth"]

            self.n_estimators = self.rf_best_params["n_estimators"]

            self.log_writer.log(
                table_name=self.table_name,
                log_info=f"{self.xgb_model} model best params are {self.xgb_best_params}",
            )

            xgb_model = XGBRegressor(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
            )

            self.log_writer.log(
                table_name=self.table_name,
                log_info=f"Initialized {self.xgb_model_name} model with best params as {self.xgb_best_params}",
            )

            xgb_model.fit(train_x, train_y)

            self.log_writer.log(
                table_name=self.table_name,
                log_info=f"Created {self.xgb_model_name} model with best params as {self.xgb_best_params}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return xgb_model

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_trained_models
        Description :   Find out the Model which has the best score.
        Output      :   The best model name and the model objectect
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_trained_models.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.table_name,
        )

        try:
            xgb_model = self.get_best_params_for_xgboost(train_x, train_y)

            xgb_model_score = self.model_utils.get_model_score(
                model=xgb_model,
                test_x=test_x,
                test_y=test_y,
                table_name=self.table_name,
            )

            rf_model = self.get_best_model_for_random_forest(train_x, train_y)

            rf_model_score = self.model_utils.get_model_score(
                model=rf_model,
                test_x=test_x,
                test_y=test_y,
                table_name=self.table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.table_name,
            )

            return xgb_model, xgb_model_score, rf_model, rf_model_score

        except Exception as e:
            self.log_writer.exception_log(
                error=e, class_name=self.class_name, method_name=method_name,
            )
