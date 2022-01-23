import os

import pandas as pd
from utils.logger import App_Logger
from utils.main_utils import read_params


class data_transform_pred:
    """
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None

    """

    def __init__(self):
        self.config = read_params()

        self.goodDataPath = self.config["data"]["good"]["pred"]

        self.log_writter = App_Logger()

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_data_transform_log = self.config["pred_db_log"]["data_transform_log"]

    def addQuotesToStringValuesInColumn(self):

        """
        Method Name: addQuotesToStringValuesInColumn
        Description: This method replaces the missing values in columns with "NULL" to
                     store in the table. We are using substring in the first column to
                     keep only "Integer" data for ease up the loading.
                     This column is anyways going to be removed during prediction.

         Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        try:
            log_file = open("Prediction_Logs/dataTransformLog.txt", "a+")

            onlyfiles = [f for f in os.listdir(self.goodDataPath)]

            for file in onlyfiles:
                f = os.path.join(self.goodDataPath, file)

                data = pd.read_csv(f)

                data["DATE"] = data["DATE"].apply(lambda x: "'" + str(x) + "'")

                csv_file = os.path.join(self.goodDataPath, file)

                data.to_csv(csv_file, index=None, header=True)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.pred_data_transform_log,
                    log_message=" %s: Quotes added successfully!!" % file,
                )

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.pred_data_transform_log,
                log_message=f"Exception occured in Class : dataTransformPredict, Method : addQuotesToStringValuesInColumn, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : dataTransformPredict, Method : addQuotesToStringValuesInColumn, Error : ",
                str(e),
            )
