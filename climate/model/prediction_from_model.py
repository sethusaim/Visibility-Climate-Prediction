import pandas as pd
from climate.data_ingestion.data_loader_prediction import data_getter_pred
from climate.data_preprocessing.preprocessing import Preprocessor
from climate.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.read_params import read_params


class prediction:
    def __init__(self):

        self.config = read_params()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.model_bucket = self.config["s3_bucket"]["scania_model_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["inputs_files_bucket"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.pred_output_file = self.config["pred_output_file"]

        self.log_writer = App_Logger()

        self.s3_obj = S3_Operations()

        self.data_getter_pred = data_getter_pred(
            db_name=self.db_name, collection_name=self.pred_log
        )

        self.preprocessor = Preprocessor(
            db_name=self.db_name, collection_name=self.pred_log
        )

        self.class_name = self.__class__.__name__

    def prediction_from_model(self):
        method_name = self.prediction_from_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.pred_log,
        )

        try:
            self.s3_obj.delete_pred_file(
                db_name=self.db_name, collection_name=self.pred_log
            )

            data = self.data_getter_pred.get_data()

            data = self.preprocessor.dropUnnecessaryColumns(
                data,
                ["DATE", "Precip", "WETBULBTEMPF", "DewPointTempF", "StationPressure"],
            )

            data = self.preprocessor.replaceInvalidValuesWithNull(data)

            is_null_present = self.preprocessor.is_null_present(data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data)

            data_scaled = pd.DataFrame(
                self.preprocessor.standardScalingData(data), columns=data.columns
            )

            kmeans = self.s3_obj.load_model_from_s3(
                bucket=self.model_bucket,
                model_name="KMeans",
                db_name=self.db_name,
                collection_name=self.pred_log,
            )

            clusters = kmeans.predict(data_scaled)

            data_scaled["clusters"] = clusters

            clusters = data_scaled["clusters"].unique()

            result = []

            for i in clusters:
                cluster_data = data_scaled[data_scaled["clusters"] == i]

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                model_name = self.s3_obj.find_correct_model_file(
                    cluster_number=i,
                    bucket_name=self.model_bucket,
                    db_name=self.db_name,
                    collection_name=self.pred_log,
                )

                model = self.s3_obj.load_model_from_s3(
                    bucket=self.model_bucket,
                    model_name=model_name,
                    db_name=self.db_name,
                    collection_name=self.pred_log,
                )

                for val in model.predict(cluster_data.values):
                    result.append(val)

            result = pd.DataFrame(result, columns=["Predictions"])

            self.s3_obj.upload_df_as_csv_to_s3(
                data_frame=result,
                file_name=self.pred_output_file,
                bucket=self.input_files_bucket,
                dest_file_name=self.pred_output_file,
                db_name=self.db_name,
                collection_name=self.pred_log,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_log,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_log,
                log_message="End of Prediction",
            )

            return (
                self.input_files_bucket,
                self.pred_output_file,
                result.head().to_json(orient="records"),
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_log,
            )
