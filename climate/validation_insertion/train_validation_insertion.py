from src.dataTransform.data_transformation_train import dataTransform
from src.dataTypeValid.data_type_valid_train import dBOperation
from src.raw_data_validation.train_data_validation import Raw_Data_validation
from utils.logger import App_Logger
from utils.read_params import read_params


class train_validation:
    def __init__(self, path):
        self.config = read_params()

        self.raw_data = Raw_Data_validation(path)

        self.dataTransform = dataTransform()

        self.dBOperation = dBOperation()

        self.db_name = self.config["db_log"]["db_train_log"]

        self.train_main_log = self.config["train_db_log"]["training_main_log"]

        self.log_writer = App_Logger()

    def train_validation(self):
        try:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Start of Validation on files for training!!",
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
                collection_name=self.train_main_log,
                log_message="Raw Data Validation Complete!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Starting Data Transforamtion!!",
            )

            self.dataTransform.addQuotesToStringValuesInColumn()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="DataTransformation Completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Creating Training_Database and tables on the basis of given schema!!!",
            )

            self.dBOperation.createTableDb(
                self.config["db_name"]["train_db_name"], column_names
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Table creation Completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Insertion of Data into Table started!!!!",
            )

            self.dBOperation.insertIntoTableGoodData(
                self.config["db_name"]["train_db_name"]
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Insertion in Table completed!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Deleting Good Data Folder!!!",
            )

            self.raw_data.deleteExistingGoodDataTrainingFolder()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Good_Data folder deleted!!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Moving bad files to Archive and deleting Bad_Data folder!!!",
            )

            self.raw_data.moveBadFilesToArchiveBad()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Bad files moved to archive!! Bad folder Deleted!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Validation Operation completed!!",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.train_main_log,
                log_message="Extracting csv file from table",
            )

            self.dBOperation.selectingDatafromtableintocsv(
                Database=self.config["db_name"]["train_db_name"]
            )

        except Exception as e:
            raise e
