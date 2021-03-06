from climate.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Data_Transform_Train:
    """
    Description :  This class shall be used for transforming the trainiction batch data before loading it in Database!!.

    Version     :   1.2
    Revisions   :   None
    """

    def __init__(self):
        self.config = read_params()

        self.train_data_bucket = self.config["s3_bucket"]["climate_train_data"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.good_train_data_dir = self.config["data"]["train"]["good"]

        self.class_name = self.__class__.__name__

        self.train_data_transform_log = self.config["train_db_log"]["data_transform"]

    def add_quotes_string(self):
        """
        Method Name :   add_quotes_string
        Description :   This method addes the quotes to the string data present in columns

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.add_quotes_string.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.train_data_transform_log,
        )

        try:
            lst = self.s3.read_csv_folder(
                self.good_train_data_dir,
                self.train_data_bucket,
                self.train_data_transform_log,
            )

            for idx, f in enumerate(lst):
                df = f[idx][0]

                file = f[idx][1]

                abs_f = f[idx][2]

                if file.endswith(".csv"):
                    df["DATE"] = df["DATE"].apply(lambda x: "'" + str(x) + "'")

                    self.log_writer.log(
                        self.train_data_transform_log,
                        f"Quotes added for the file {file}",
                    )

                    self.s3.upload_df_as_csv(
                        df,
                        abs_f,
                        file,
                        self.train_data_bucket,
                        self.train_data_transform_log,
                    )

                else:
                    pass

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.train_data_transform_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.train_data_transform_log,
            )
