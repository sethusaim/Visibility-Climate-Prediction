import os

import pandas as pd
from utils.logger import App_Logger
from utils.read_params import read_params


class data_transform_train:
    """
    Written By  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None

    """

    def __init__(self):
        self.config = read_params()

        self.goodDataPath = self.config["data"]["good"]["train"]

        self.logger = App_Logger()

        self.db_name = self.config["db_log"]["db_train_log"]

        self.addQuotesToString_log = self.config["train_db_log"][
            "addQuotesToString_log"
        ]

    def addQuotesToStringValuesInColumn(self):
        """
        Method Name :   addQuotesToStringValuesInColumn
        Description :   This method converts all the columns with string datatype such that
                        each value for that column is enclosed in quotes. This is done
                        to avoid the error while inserting string values in table as varchar.
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        try:
            onlyfiles = [f for f in os.listdir(self.goodDataPath)]

            for file in onlyfiles:
                f = os.path.join(self.goodDataPath, file)

                data = pd.read_csv(f)

                data["DATE"] = data["DATE"].apply(lambda x: "'" + str(x) + "'")

                data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)

                self.logger.log(
                    db_name=self.db_name,
                    collection_name=self.addQuotesToString_log,
                    log_message=" %s:  Quotes added successfully!!" % file,
                )

        except Exception as e:
            self.logger.log(
                db_name=self.db_name,
                collection_name=self.addQuotesToString_log,
                log_message=f"Exception occured in Class : dataTransform \
                    Method : addQuotesToStringValuesInColumn, Error : {str(e)}",
            )
