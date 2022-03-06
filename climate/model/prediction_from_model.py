import pandas as pd
from botocore.exceptions import ClientError
from sklearn.neighbors import LocalOutlierFactor
from climate.data_ingestion.data_loader_prediction import Data_Getter_Pred
from climate.data_preprocessing.preprocessing import Preprocessor
from climate.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Prediction:
    """
    Description :   This class shall be used for loading the production model

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.model_bucket = self.config["s3_bucket"]["climate_model_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["inputs_files_bucket"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.pred_output_file = self.config["pred_output_file"]

        self.log_writer = App_Logger()

        self.s3 = S3_Operation()

        self.data_getter_pred = Data_Getter_Pred(table_name=self.pred_log)

        self.preprocessor = Preprocessor(table_name=self.pred_log)

        self.class_name = self.__class__.__name__

    def delete_pred_file(self, table_name):
        """
        Method Name :   delete_pred_file
        Description :   This method is used for deleting the existing prediction batch file

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_pred_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.s3.load_objectect(
                bucket_name=self.input_files_bucket, object=self.pred_output_file
            )

            self.log_writer.log(
                table_name=table_name,
                log_info=f"Found existing prediction batch file. Deleting it.",
            )

            self.s3.delete_file(
                bucket_name=self.input_files_bucket,
                file=self.pred_output_file,
                table_name=table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pass

            else:
                self.log_writer.exception_log(
                    error=e,
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

    def find_correct_model_file(self, cluster_number, bucket_name, table_name):
        """
        Method Name :   find_correct_model_file
        Description :   This method is used for finding the correct model file during prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.find_correct_model_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            list_of_files = self.s3.get_files(
                bucket=bucket_name,
                folder_name=self.prod_model_dir,
                table_name=table_name,
            )

            for file in list_of_files:
                try:
                    if file.index(str(cluster_number)) != -1:
                        model_name = file

                except:
                    continue

            model_name = model_name.split(".")[0]

            self.log_writer.log(
                table_name=table_name,
                log_info=f"Got {model_name} from {self.prod_model_dir} folder in {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return model_name

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def predict_from_model(self):
        """
        Method Name :   predict_from_model
        Description :   This method is used for loading from prod model dir of s3 bucket and use them for prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.predict_from_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=self.pred_log,
        )

        try:
            self.delete_pred_file(table_name=self.pred_log)

            data = self.data_getter_pred.get_data()

            is_null_present = self.preprocessor.is_null_present(data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data)

            cols_to_drop = self.preprocessor.get_columns_with_zero_std_deviation(data)

            data = self.preprocessor.remove_columns(data, cols_to_drop)

            kmeans = self.s3.load_model(
                bucket=self.model_bucket, model_name="KMeans", table_name=self.pred_log
            )

            clusters = kmeans.predict(data.drop(["climate"], axis=1))

            data["clusters"] = clusters

            clusters = data["clusters"].unique()

            for i in clusters:
                cluster_data = data[data["clusters"] == i]

                climate_names = list(cluster_data["climate"])

                cluster_data = data.drop(labels=["climate"], axis=1)

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                crt_model_name = self.find_correct_model_file(
                    cluster_number=i,
                    bucket_name=self.model_bucket,
                    table_name=self.pred_log,
                )

                model = self.s3.load_model(model_name=crt_model_name)

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(
                    list(zip(climate_names, result)), columns=["climate", "Prediction"]
                )

                self.s3.upload_df_as_csv(
                    data_frame=result,
                    local_file_name=self.pred_output_file,
                    bucket_file_name=self.pred_output_file,
                    bucket_name=self.input_files_bucket,
                    table_name=self.pred_log,
                )

            self.log_writer.log(
                table_name=self.pred_log, log_info=f"End of Prediction"
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_log,
            )

            return (
                self.input_files_bucket,
                self.pred_output_file,
                result.head().to_json(orient="records"),
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=self.pred_log,
            )
