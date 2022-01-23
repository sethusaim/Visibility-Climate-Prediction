from climate.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params


class data_transform_pred:
    """
    Description :  This class shall be used for transforming the training batch data before loading it in Database!!.
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.pred_data_bucket = self.config["s3_bucket"]["climate_pred_data_bucket"]

        self.s3_obj = S3_Operations()

        self.log_writer = App_Logger()

        self.good_pred_data_dir = self.config["data"]["pred"]["good_data_dir"]

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform"]

    def add_quotes_to_string(self):
        """
        Method Name :   add_quotes_to_string
        Description :   This method addes the quotes to the string data present in columns
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.add_quotes_to_string.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.pred_data_transform_log,
        )

        try:
            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.pred_data_bucket,
                filename=self.good_pred_data_dir,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                    df["DATE"] = df["DATE"].apply(lambda x: "'" + str(x) + "'")

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                        log_message=f"Quotes added for the file {file}",
                    )

                    self.s3_obj.upload_df_as_csv_to_s3(
                        data_frame=df,
                        file_name=abs_f,
                        bucket=self.pred_data_bucket,
                        dest_file_name=file,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )


from climate.s3_bucket_operations.s3_operations import S3_Operations
from utils.logger import App_Logger
from utils.main_utils import convert_object_to_dataframe
from utils.read_params import read_params


class data_transform_pred:
    """
    Description :  This class shall be used for transforming the preding batch data before loading it in Database!!.
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.pred_data_bucket = self.config["s3_bucket"]["climate_pred_data_bucket"]

        self.s3_obj = S3_Operations()

        self.log_writer = App_Logger()

        self.good_pred_data_dir = self.config["data"]["pred"]["good_data_dir"]

        self.class_name = self.__class__.__name__

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform"]

    def add_quotes_to_string(self):
        """
        Method Name :   add_quotes_to_string
        Description :   This method addes the quotes to the string data present in columns
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.add_quotes_to_string.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.pred_data_transform_log,
        )

        try:
            csv_file_objs = self.s3_obj.get_file_objects_from_s3(
                bucket=self.pred_data_bucket,
                filename=self.good_pred_data_dir,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

            for f in csv_file_objs:
                file = f.key

                abs_f = file.split("/")[-1]

                if file.endswith(".csv"):
                    df = convert_object_to_dataframe(
                        obj=f,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                    df["DATE"] = df["DATE"].apply(lambda x: "'" + str(x) + "'")

                    self.log_writer.log(
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                        log_message=f"Quotes added for the file {file}",
                    )

                    self.s3_obj.upload_df_as_csv_to_s3(
                        data_frame=df,
                        file_name=abs_f,
                        bucket=self.pred_data_bucket,
                        dest_file_name=file,
                        db_name=self.db_name,
                        collection_name=self.pred_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
            )
