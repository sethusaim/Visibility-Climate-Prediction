from climate.data_transform.data_transformation_pred import data_transform_pred
from climate.data_type_valid.data_type_valid_pred import db_operation_pred
from climate.raw_data_validation.pred_data_validation import Prediction_Data_validation
from utils.logger import App_Logger
from utils.main_utils import read_params


class pred_validation:
    def __init__(self, path):
        self.config = read_params()

        self.raw_data = Prediction_Data_validation(path)

        self.data_transform = data_transform_pred()

        self.db_operation = db_operation_pred()

        self.file_object = open("Prediction_Logs/Prediction_Log.txt", "a+")

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_main_log = self.config["pred_db_log"]["pred_main"]

        self.log_writer = App_Logger()

    def prediction_validation(self):

        try:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Start of Validation on files for prediction!!",
            )

            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                noofcolumns,
            ) = self.raw_data.valuesFromSchema()

            regex = self.raw_data.manualRegexCreation()

            self.raw_data.validationFileNameRaw(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validateColumnLength(noofcolumns)

            self.raw_data.validateMissingValuesInWholeColumn()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Raw Data Validation Complete!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Starting Data Transforamtion!!",
            )

            self.dataTransform.addQuotesToStringValuesInColumn()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="DataTransformation Completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Creating Prediction_Database and tables on the basis of given schema!!!",
            )

            self.dBOperation.createTableDb(
                self.config["db_names"]["pred_db_name"], column_names
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Table creation Completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Insertion of Data into Table started!!!!",
            )

            self.dBOperation.insertIntoTableGoodData(
                self.config["db_name"]["pred_db_name"]
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Insertion in Table completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Deleting Good Data Folder!!!",
            )

            self.raw_data.deleteExistingGoodDataTrainingFolder()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Good_Data folder deleted!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Moving bad files to Archive and deleting Bad_Data folder!!!",
            )

            self.raw_data.moveBadFilesToArchiveBad()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Bad files moved to archive!! Bad folder Deleted!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Validation Operation completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message="Extracting csv file from table",
            )

            self.dBOperation.selectingDatafromtableintocsv(
                self.config["db_names"]["pred_db_name"]
            )

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_main_log,
                log_message=f"Exception cccured in Class : pred_validation, Method : prediction_validation, Error : {str(e)}",
            )

            raise Exception(
                f"Exception cccured in Class : pred_validation, Method : prediction_validation, Error : ",
                str(e),
            )
