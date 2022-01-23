import mlflow
from sklearn.model_selection import train_test_split
from src.data_ingestion.data_loader_train import Data_Getter
from src.data_preprocessing.clustering import KMeansClustering
from src.data_preprocessing.preprocessing import Preprocessor
from src.file_operations.file_methods import File_Operation
from src.model_finder.tuner import Model_Finder
from utils.logger import App_Logger
from utils.main_utils import (
    log_metric_to_mlflow,
    log_model_to_mlflow,
    log_param_to_mlflow,
)
from utils.read_params import read_params


class trainModel:
    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.db_name = self.config["db_log"]["db_train_log"]

        self.model_train_log = self.config["train_db_log"]["model_training_log"]

    def trainingModel(self):
        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.model_train_log,
            log_message="Start of Training",
        )

        try:
            data_getter = Data_Getter(
                db_name=self.db_name, collection_name=self.model_train_log
            )

            data = data_getter.get_data()

            preprocessor = Preprocessor(
                db_name=self.db_name, collection_name=self.model_train_log
            )

            data = preprocessor.dropUnnecessaryColumns(
                data,
                ["DATE", "Precip", "WETBULBTEMPF", "DewPointTempF", "StationPressure"],
            )

            data = preprocessor.replaceInvalidValuesWithNull(data)

            is_null_present = preprocessor.is_null_present(data)

            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            X, Y = preprocessor.separate_label_feature(
                data, label_column_name="VISIBILITY"
            )

            kmeans = KMeansClustering(
                db_name=self.db_name, collection_name=self.model_train_log
            )

            number_of_clusters = kmeans.elbow_plot(X)

            X, kmeans_model = kmeans.create_clusters(X, number_of_clusters)

            X["Labels"] = Y

            list_of_clusters = X["Cluster"].unique()

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]

                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)

                cluster_label = cluster_data["Labels"]

                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features,
                    cluster_label,
                    test_size=self.config["base"]["test_size"],
                    random_state=self.config["base"]["random_state"],
                )

                x_train_scaled = preprocessor.standardScalingData(x_train)

                x_test_scaled = preprocessor.standardScalingData(x_test)

                model_finder = Model_Finder(
                    db_name=self.db_name, collection_name=self.model_train_log
                )

                (
                    dt_model_error,
                    dt_model,
                    xgb_model_error,
                    xgb_model,
                    rf_model_error,
                    rf_model,
                ) = model_finder.get_trained_models(
                    x_train_scaled, y_train, x_test_scaled, y_test
                )

                file_op = File_Operation(
                    db_name=self.db_name, collection_name=self.model_train_log
                )

                saved_dt_model = file_op.save_model(
                    model=dt_model,
                    filename=self.config["model_names"]["dt_model_name"] + str(i),
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message="Saved "
                    + self.config["model_names"]["dt_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                saved_rf_model = file_op.save_model(
                    model=rf_model,
                    filename=self.config["model_names"]["rf_model_name"] + str(i),
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message="Saved "
                    + self.config["model_names"]["rf_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                saved_xgb_model = file_op.save_model(
                    model=xgb_model,
                    filename=self.config["model_names"]["xgb_model_name"] + str(i),
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.model_train_log,
                    log_message="Saved "
                    + self.config["model_names"]["xgb_model_name"]
                    + str(i)
                    + " in trained model folder",
                )

                try:
                    remote_server_uri = self.config["mlflow_config"][
                        "remote_server_uri"
                    ]

                    mlflow.set_tracking_uri(remote_server_uri)

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message="Set the remote server uri",
                    )

                    mlflow.set_experiment(
                        experiment_name=self.config["mlflow_config"]["experiment_name"]
                    )

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message=f"Experiment name has been set to {self.config['mlflow_config']['experiment_name']}",
                    )

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message="Started mlflow server with "
                        + self.config["mlflow_config"]["run_name"],
                    )

                    with mlflow.start_run(
                        run_name=self.config["mlflow_config"]["run_name"]
                    ):
                        ##### KMeans model #####

                        log_model_to_mlflow(
                            model=kmeans_model,
                            model_name=self.config["model_names"]["kmeans_model_name"],
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        ###### XGBoost ##########

                        log_param_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["xgb_model_name"]
                            + str(i),
                            param_name="learning rate",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["xgb_model_name"]
                            + str(i),
                            param_name="max_depth",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["xgb_model_names"]
                            + str(i),
                            param_name="n_estimators",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_metric_to_mlflow(
                            model_name=self.config["model_names"]["xgb_model_name"]
                            + str(i),
                            metric=xgb_model_error,
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_model_to_mlflow(
                            model=xgb_model,
                            model_name=self.config["model_names"]["xgb_model_name"],
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        ###### Random Forest ####

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="criterion",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="max_depth",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="n_estimators",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            param_name="max_features",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_metric_to_mlflow(
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            metric=rf_model_error,
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_model_to_mlflow(
                            model=rf_model,
                            model_name=self.config["model_names"]["rf_model_name"]
                            + str(i),
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        ## Decision Tree ########

                        log_param_to_mlflow(
                            model=dt_model,
                            model_name=self.config["model_names"]["dt_model_name"]
                            + str(i),
                            param_name="criterion",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=dt_model,
                            model_name=self.config["model_names"]["dt_model_name"]
                            + str(i),
                            param_name="splitter",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=dt_model,
                            model_name=self.config["model_names"]["dt_model_name"]
                            + str(i),
                            param_name="max_features",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            model=dt_model,
                            model_name=self.config["model_names"]["dt_model_name"],
                            param_name="max_depth",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_param_to_mlflow(
                            mode=dt_model,
                            model_name=self.config["model_names"]["dt_model_name"]
                            + str(i),
                            param_name="min_samples_split",
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_metric_to_mlflow(
                            model_name=self.config["model_names"]["dt_model_name"]
                            + str(i),
                            metric=dt_model_error,
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                        log_model_to_mlflow(
                            model=dt_model,
                            model_name=self.config["model_names"]["dt_model_name"]
                            + str(i),
                            db_name=self.db_name,
                            collection_name=self.model_train_log,
                        )

                except Exception as e:
                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.model_train_log,
                        log_message=f"Exception Occured in Class : trainModel, Method : mlflow , Error : {str(e)}",
                    )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message="Successful End of Training",
            )

            return number_of_clusters

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.model_train_log,
                log_message=f"Exception occured in Class : trainModel, Method : trainingModel, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : trainModel, Method : trainingModel, Error : ",
                str(e),
            )
