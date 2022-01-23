import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from utils.logger import App_Logger
from utils.read_params import read_params


class Preprocessor:
    """
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None

    """

    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.db_name = db_name

        self.collection_name = collection_name

        self.log_writter = App_Logger()

    def remove_columns(self, data, columns):
        """
        Method Name: remove_columns
        Description: This method removes the given columns from a pandas dataframe.
        Output: A pandas DataFrame after removing the specified columns.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the remove_columns method of the Preprocessor class",
        )

        self.data = data

        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column removal Successful.Exited the remove_columns method of the Preprocessor class",
            )

            return self.useful_data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : remove_columns, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : remove_columns, Error : ",
                str(e),
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the features and a Label Coulmns.
        Output      :   Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the separate_label_feature method of the Preprocessor class",
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class",
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : separate_label_feature, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : separate_label_feature, Error : ",
                str(e),
            )

    def dropUnnecessaryColumns(self, data, columnNameList):
        """
        Method Name :   is_null_present
        Description :   This method drops the unwanted columns as discussed in EDA section.
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        try:
            data = data.drop(columnNameList, axis=1)

            return data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : dropUnnecessaryColumns, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor, Method : dropUnnecessaryColumns, Error : ",
                str(e),
            )

    def replaceInvalidValuesWithNull(self, data):

        """
        Method Name :   is_null_present
        Description :   This method replaces invalid values i.e. '?' with null, as discussed in EDA.
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        try:
            for column in data.columns:
                count = data[column][data[column] == "?"].count()

                if count != 0:
                    data[column] = data[column].replace("?", np.nan)

            return data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor. \
                    Method : replaceInvalidValuesWithNull, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : Preprocessor. \
                Method : replaceInvalidValuesWithNull, Error : ",
                str(e),
            )

    def is_null_present(self, data):
        """
        Method Name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas Dataframe or not.
        Output      :   Returns True if null values are present in the DataFrame, False if they are not present and
                        returns the list of columns for which null values are present.
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the is_null_present method of the Preprocessor class",
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
                self.dataframe_with_null = pd.DataFrame()

                self.dataframe_with_null["columns"] = data.columns

                self.dataframe_with_null["missing values count"] = np.asarray(
                    data.isna().sum()
                )

                self.dataframe_with_null.to_csv("preprocessing_data/null_values.csv")

                self.dataframe_with_null.to_csv(self.config["null_values_csv_file"])

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values is a success.Data written to the null values file.\
                     Exited the is_null_present method of the Preprocessor class",
            )

            return self.null_present

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : is_null_present, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values failed. Exited the is_null_present method of the Preprocessor class",
            )

            raise Exception(
                f"Exception occured in Class : Preprocessor, Method : is_null_present, Error : {str(e)}"
            )

    def encodeCategoricalValues(self, data):
        """
        Method Name :   encodeCategoricalValues
        Description :   This method encodes all the categorical values in the training set.
        Output      :   A Dataframe which has all the categorical values encoded.
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None
        """
        try:
            data["class"] = data["class"].map({"p": 1, "e": 2})

            for column in data.drop(["class"], axis=1).columns:
                data = pd.get_dummies(data, columns=[column])

            return data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values failed. \
                    Exited the encodeCategoricalValues method of the Preprocessor class",
            )

            raise Exception(
                f"Exception occured in Class : Preprocessor. \
                Method : encodeCategoricalValues, Error : {str(e)}"
            )

    def encodeCategoricalValuesPrediction(self, data):
        """
        Method Name :   encodeCategoricalValuesPrediction
        Description :   This method encodes all the categorical values in the prediction set.
        Output      :   A Dataframe which has all the categorical values encoded.
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None
        """
        try:
            for column in data.columns:
                data = pd.get_dummies(data, columns=[column])

            return data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values failed. \
                    Exited the encodeCategoricalValues method of the Preprocessor class",
            )

            raise Exception(
                f"Exception occured in Class : Preprocessor. \
                Method : encodeCategoricalValues, Error : {str(e)}"
            )

    def standardScalingData(self, X):
        try:
            scalar = StandardScaler()

            X_scaled = scalar.fit_transform(X)

            return X_scaled

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding missing values failed. \
                    Exited the standardScalingData method of the Preprocessor class",
            )

            raise Exception(
                f"Exception occured in Class : Preprocessor. \
                Method : standardScalingData, Error : {str(e)}"
            )

    def impute_missing_values(self, data):
        """
        Method Name: impute_missing_values
        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
        Output: A Dataframe which has all the missing values imputed.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the impute_missing_values method of the Preprocessor class",
        )

        self.data = data

        try:
            imputer = KNNImputer(
                n_neighbors=self.config["n_neighbors"],
                weights=self.config["weights"],
                missing_values=np.nan,
            )

            self.new_array = imputer.fit_transform(self.data)

            self.new_data = pd.DataFrame(
                data=(self.new_array), columns=self.data.columns
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Imputing missing values Successful. \
                    Exited the impute_missing_values method of the Preprocessor class",
            )

            return self.new_data

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor, Method : impute_missing_values, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Imputing missing values failed. \
                    Exited the impute_missing_values method of the Preprocessor class",
            )

            raise Exception(
                f"Exception occured in Class : Preprocessor. \
                Method : impute_missing_values, Error : {str(e)}"
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_columns_with_zero_std_deviation
        Description :   This method finds out the columns which have a standard deviation of zero.
        Output      :   List of the columns with standard deviation of zero
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the get_columns_with_zero_std_deviation method of the Preprocessor class",
        )

        self.columns = data.columns

        self.data_n = data.describe()

        self.col_to_drop = []

        try:
            for x in self.columns:
                if self.data_n[x]["std"] == 0:
                    self.col_to_drop.append(x)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column search for Standard Deviation of Zero Successful. \
                    Exited the get_columns_with_zero_std_deviation method of the Preprocessor class",
            )

            return self.col_to_drop

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : Preprocessor. \
                    Method : get_columns_with_zero_std_deviation, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Column search for Standard Deviation of Zero Failed. \
                    Exited the get_columns_with_zero_std_deviation method of the Preprocessor class",
            )

            raise Exception(
                f"Exception occured in Class : Preprocessor. \
                Method : get_columns_with_zero_std_deviation, Error : {str(e)}"
            )
