import numpy as np
import pandas as pd
from climate.s3_bucket_operations.s3_operations import S3_Operation
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from utils.logger import App_Logger
from utils.read_params import read_params


class Preprocessor:
    """
    Written By  :   iNeuron Intelligence
    Version     :   1.2
    Revisions   :   moved setup to cloud
    """

    def __init__(self, log_file):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_file = log_file

        self.null_values_file = self.config["null_values_csv_file"]

        self.n_components = self.config["pca_model"]["n_components"]

        self.knn_n_neighbors = self.config["n_neighbors"]

        self.knn_weights = (self.config["weights"],)

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.s3 = S3_Operation()

    def remove_columns(self, data, columns):
        """
        Method Name :   remove_columns
        Description :   This method removes the given columns from a pandas dataframe.

        Output      :   A pandas dataframe after removing the specified columns.
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.remove_columns.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        self.data = data

        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writer.log(self.log_file, "Column removal Successful")

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return self.useful_data

        except Exception as e:
            self.log_writer.log(self.log_file, "Column removal Unsuccessful")

            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the features and a Label Coulmns.

        Output      :   Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.separate_label_feature.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writer.log(
                self.log_file,
                "Label Separation Successful",
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writer.log(
                self.log_file,
                "Label Separation Unsuccessful",
            )

            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def drop_unnecessary_columns(self, data, cols):
        """
        Method Name :   drop_unnecessary_columns
        Description :   This method drop unnecessary columns in the dataframe

        Output      :   Unnecessary columns are dropped in the dataframe
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.drop_unnecessary_columns.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        try:
            data = data.drop(cols, axis=1)

            self.log_writer.log(self.log_file, "Dropped unnecessary columns")

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def replace_invalid_with_null(self, data):
        """
        Method Name :   replace_invalid_with_null
        Description :   This method replaces invalid values with null

        Output      :   A dataframe where invalid values are replaced with null
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.replace_invalid_with_null.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        try:
            for column in data.columns:
                count = data[column][data[column] == "?"].count()

                if count != 0:
                    data[column] = data[column].replace("?", np.nan)

            self.log_writer.log(
                self.log_file,
                "Replaced invalid values with np.nan",
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def is_null_present(self, data):
        """
        Method Name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas dataframe or not.

        Output      :   Returns True if null values are present in the dataframe, False if they are not present and
                        returns the list of columns for which null values are present.
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.is_null_present.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        self.null_present = False

        self.cols_with_missing_values = []

        self.cols = data.columns

        try:
            self.null_counts = data.isna().sum()

            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True

                    self.cols_with_missing_values.append(self.cols[i])

            if self.null_present:
                self.null_df = pd.DataFrame()

                self.null_df["columns"] = data.columns

                self.null_df["missing values count"] = np.asarray(data.isna().sum())

            self.log_writer.log(self.log_file, "Created data frame with null values")

            self.s3.upload_df_as_csv(
                self.null_df,
                self.null_values_file,
                self.input_files_bucket,
                self.null_values_file,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return self.null_present

        except Exception as e:
            self.log_writer.log(self.log_file, "Finding missing values failed")

            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def encode_target_cols(self, data):
        """
        Method Name :   encode_target_cols
        Description :   This method encodes all the categorical values in the training set.

        Output      :   A dataframe which has all the categorical values encoded.
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.encode_target_cols.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        try:
            data["class"] = data["class"].map({"p": 1, "e": 2})

            for column in data.drop(["class"], axis=1).columns:
                data = pd.get_dummies(data, columns=[column])

            self.log_writer.log(self.log_file, "Encoded target columns")

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def apply_standard_scaler(self, X):
        """
        Method Name : apply_standard_scaler
        Description : This method replaces all the missing values in the dataframe using KNN Imputer.

        Output      : A dataframe which has all the missing values imputed.
        On Failure  : Raise Exception

        Written By  : iNeuron Intelligence
        Version     : 1.2
        Revisions   : moved setup to cloud
        """
        method_name = self.apply_standard_scaler.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        try:
            scalar = StandardScaler()

            X_scaled = scalar.fit_transform(X)

            self.log_writer.log(
                self.log_file,
                f"Transformed data using {scalar.__class__.__name__}",
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return X_scaled

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def impute_missing_values(self, data):
        """
        Method Name : impute_missing_values
        Description : This method replaces all the missing values in the dataframe using KNN Imputer.

        Output      : A dataframe which has all the missing values imputed.
        On Failure  : Raise Exception

        Written By  : iNeuron Intelligence
        Version     : 1.2
        Revisions   : moved setup to cloud
        """
        method_name = self.impute_missing_values.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        self.data = data

        try:
            imputer = KNNImputer(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                missing_values=np.nan,
            )

            self.log_writer.log(
                self.log_file,
                f"Initialized {imputer.__class__.__name__}",
            )

            self.new_array = imputer.fit_transform(self.data)

            self.log_writer.log(
                self.log_file,
                "Imputed missing values using KNN imputer",
            )

            self.new_data = pd.dataframe(
                data=(self.new_array), columns=self.data.columns
            )

            self.log_writer.log(
                self.log_file,
                "Created new dataframe with imputed values",
            )

            self.log_writer.log(
                self.log_file,
                "Imputing missing values Successful",
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return self.new_data

        except Exception as e:
            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_columns_with_zero_std_deviation
        Description :   This method finds out the columns which have a standard deviation of zero.

        Output      :   List of the columns with standard deviation of zero
        On Failure  :   Write an exception log and then raise an exception

        Written By  :   iNeuron Intelligence
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_columns_with_zero_std_deviation.__name__

        self.log_writer.start_log(
            "start",
            self.class_name,
            method_name,
            self.log_file,
        )

        self.columns = data.columns

        self.data_n = data.describe()

        self.col_drop = []

        try:
            for x in self.columns:
                if self.data_n[x]["std"] == 0:
                    self.col_drop.append(x)

            self.log_writer.log(
                self.log_file,
                "Column search for Standard Deviation of Zero Successful",
            )

            self.log_writer.start_log(
                "exit",
                self.class_name,
                method_name,
                self.log_file,
            )

            return self.col_drop

        except Exception as e:
            self.log_writer.log(
                self.log_file,
                "Column search for Standard Deviation of Zero Failed",
            )

            self.log_writer.exception_log(
                e,
                self.class_name,
                method_name,
                self.log_file,
            )
