import pandas as pd
from src.data_ingestion.data_loader_prediction import Data_Getter_Pred
from src.data_preprocessing.preprocessing import Preprocessor
from src.file_operations.file_methods import File_Operation
from src.raw_data_validation.pred_data_validation import Prediction_Data_validation
from utils.logger import App_Logger
from utils.main_utils import read_params


class prediction:
    def __init__(self, path):
        self.config = read_params()

        self.db_name = self.config["db_log"]["db_pred_log"]

        self.pred_model_log = self.config["pred_db_log"]["pred_main_log"]

        self.log_writer = App_Logger()

        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile()

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_model_log,
                log_message="Start of Prediction",
            )

            data_getter = Data_Getter_Pred(
                db_name=self.db_name, collection_name=self.pred_model_log
            )

            data = data_getter.get_data()

            preprocessor = Preprocessor(
                db_name=self.db_name, collection_name=self.pred_model_log
            )

            data = preprocessor.dropUnnecessaryColumns(
                data,
                ["DATE", "Precip", "WETBULBTEMPF", "DewPointTempF", "StationPressure"],
            )

            data = preprocessor.replaceInvalidValuesWithNull(data)

            is_null_present = preprocessor.is_null_present(data)

            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            data_scaled = pd.DataFrame(
                preprocessor.standardScalingData(data), columns=data.columns
            )

            file_op = File_Operation(
                db_name=self.db_name, collection_name=self.pred_model_log
            )

            kmeans = file_op.load_model(self.config["model_names"]["kmeans"])

            clusters = kmeans.predict(data_scaled)

            data_scaled["clusters"] = clusters

            clusters = data_scaled["clusters"].unique()

            result = []

            for i in clusters:
                cluster_data = data_scaled[data_scaled["clusters"] == i]

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                model_name = file_op.find_correct_model_file(i)

                model = file_op.load_model(model_name)

                for val in model.predict(cluster_data.values):
                    result.append(val)

            result = pd.DataFrame(result, columns=["Predictions"])

            path = self.config["pred_output_file"]

            result.to_csv(self.config["pred_output_file"], header=True)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_model_log,
                log_message="End of Prediction",
            )

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.pred_model_log,
                log_message=f"Exception occured in Class : prediction, Method : predictionFromModel, Error : {str(e)}",
            )

            raise Exception(
                "Exception occured in Class : prediction, Method : predictionFromModel, Error : ",
                str(e),
            )

        return path
