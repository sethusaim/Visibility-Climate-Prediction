import json
import os
import pickle
from io import StringIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params


class S3_Operation:
    """
    Description :   This method is used for all the S3 bucket_name operations
    
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.model_utils = Model_Utils()

        self.file_format = self.config["model_utils"]["save_format"]

    def get_s3_client(self, table_name):
        """
        Method Name :   get_s3_client
        Description :   This method gets s3 client from boto3

        Output      :   A boto3 client with s3 is created
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_s3_client.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_client = boto3.client("s3")

            self.log_writer.log(table_name=table_name, log_message="Got s3 client")

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return s3_client

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_s3_resource(self, table_name):
        """
        Method Name :   get_s3_resource
        Description :   This method gets s3 resource from boto3

        Output      :   A boto3 resource with s3 is created
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_s3_resource.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_resource = boto3.resource("s3")

            self.log_writer.log(table_name=table_name, log_message="Got s3 resource")

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return s3_resource

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_objectect(self, objectect, table_name, decode=True, make_readable=False):
        """
        Method Name :   read_objectect
        Description :   This method reads the objectect with kwargs

        Output      :   A objectect is read with kwargs
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_objectect.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            func = (
                lambda: objectect.get()["Body"].read().decode()
                if decode is True
                else objectect.get()["Body"].read()
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read the s3 objectect with decode as {decode}",
            )

            conv_func = lambda: StringIO(func()) if make_readable is True else func()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"read the s3 objectect with make_readable as {make_readable}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return conv_func()

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_text(self, file_name, bucket_name, table_name):
        """
        Method Name :   read_text
        Description :   This method reads the text data from s3 bucket

        Output      :   Text data is read from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_text.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            txt_object = self.get_file_objectect(
                file_name=file_name, bucket_name=bucket_name, table_name=table_name
            )

            content = self.read_objectect(objectect=txt_object, table_name=table_name)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read {file_name} file as text from {bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return content

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_json(self, file_name, bucket_name, table_name):
        """
        Method Name :   read_json
        Description :   This method reads the json data from s3 bucket

        Output      :   Json data is read from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_json.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            f_object = self.get_file_objectect(
                file_name=file_name, bucket_name=bucket_name, table_name=table_name
            )

            json_content = self.read_objectect(
                objectect=f_object, table_name=table_name
            )

            dic = json.loads(json_content)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Read {file_name} from {bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return dic

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_df_from_objectect(self, objectect, table_name):
        """
        Method Name :   get_df_from_objectect
        Description :   This method gets dataframe from objectect 

        Output      :   dataframe is read from the objectect
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_df_from_objectect.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            content = self.read_objectect(
                objectect=objectect, table_name=table_name, make_readable=True
            )

            df = pd.read_csv(content)

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_csv(self, file_name, bucket_name, table_name):
        """
        Method Name :   read_csv
        Description :   This method reads the csv data from s3 bucket

        Output      :   A pandas series objectect consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            csv_object = self.get_file_objectect(
                file_name=file_name, bucket_name=bucket_name, table_name=table_name,
            )

            df = self.get_df_from_objectect(objectect=csv_object, table_name=table_name)

            self.log_writer.log(
                table_name=table_name,
                log_info=f"Read {file_name} csv file from {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return df

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def read_csv_from_folder(self, folder_name, bucket_name, table_name):
        """
        Method Name :   read_csv_from_folder
        Description :   This method reads the csv files from folder

        Output      :   A list of tuple of dataframe, along with absolute file name and file name is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.read_csv_from_folder.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )
        try:
            files = self.get_files_from_folder(
                folder_name=folder_name, bucket_name=bucket_name, table_name=table_name,
            )

            lst = [
                (
                    self.read_csv(
                        file_name=f, bucket_name=bucket_name, table_name=table_name,
                    ),
                    f,
                    f.split("/")[-1],
                )
                for f in files
            ]

            self.log_writer.log(
                table_name=table_name,
                log_info=f"Read csv files from {folder_name} folder from {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return lst

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def load_objectect(self, objectect, bucket_name, table_name):
        """
        Method Name :   load_objectect
        Description :   This method loads the objectect from s3 bucket

        Output      :   An objectect is loaded from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_objectect.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.objectect(bucket_name, objectect).load()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Loaded {objectect} from {bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def create_folder(self, folder_name, bucket_name, table_name):
        """
        Method Name :   create_folder
        Description :   This method creates a folder in s3 bucket

        Output      :   A folder is created in s3 bucket 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.create_folder.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.load_objectect(bucket_name=bucket_name, objectect=folder_name)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Folder {folder_name} already exists.",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"{folder_name} folder does not exist,creating new one",
                )

                self.put_objectect(
                    objectect=folder_name,
                    bucket_name=bucket_name,
                    table_name=table_name,
                )

                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"{folder_name} folder created in {bucket_name} bucket_name",
                )

            else:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Error occured in creating {folder_name} folder",
                )

                self.log_writer.exception_log(
                    error=e,
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

    def put_objectect(self, objectect, bucket_name, table_name):
        """
        Method Name :   put_objectect
        Description :   This method puts an objectect in s3 bucket

        Output      :   An objectect is put in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.put_objectect.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_client = self.get_s3_client(table_name=table_name)

            s3_client.put_objectect(Bucket=bucket_name, Key=(objectect + "/"))

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created {objectect} folder in {bucket_name} bucket",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def upload_file(
        self, from_file_name, to_file_name, bucket_name, table_name, remove=True
    ):
        """
        Method Name :   upload_file
        Description :   This method uploades a file to s3 bucket with kwargs

        Output      :   A file is uploaded to s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.upload_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploading {from_file_name} to s3 bucket {bucket_name}",
            )

            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.meta.client.upload_file(
                from_file_name, bucket_name, to_file_name
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploaded {from_file_name} to s3 bucket {bucket_name}",
            )

            if remove is True:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Option remove is set {remove}..deleting the file",
                )

                os.remove(from_file_name)

                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Removed the local copy of {from_file_name}",
                )

                self.log_writer.start_log(
                    key="exit",
                    class_name=self.class_name,
                    method_name=method_name,
                    table_name=table_name,
                )

            else:
                self.log_writer.log(
                    table_name=table_name,
                    log_message=f"Option remove is set {remove}, not deleting the file",
                )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_bucket(self, bucket_name, table_name):
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket from s3 

        Output      :   A s3 bucket name is returned based on the bucket_name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_bucket.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_resource = self.get_s3_resource(table_name=table_name)

            bucket_name = s3_resource.Bucket(bucket_name)

            self.log_writer.log(
                table_name=table_name, log_message=f"Got {bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return bucket_name

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def copy_data(
        self, from_file_name, from_bucket_name, to_file_name, to_bucket_name, table_name
    ):
        """
        Method Name :   copy_data
        Description :   This method copies the data from one bucket to another bucket

        Output      :   The data is copied from one bucket to another
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.copy_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            copy_source = {"Bucket": from_bucket_name, "Key": from_file_name}

            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.meta.client.copy(copy_source, to_bucket_name, to_file_name)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Copied data from bucket_name {from_bucket_name} to bucket_name {to_bucket_name}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def delete_file(self, file_name, bucket_name, table_name):
        """
        Method Name :   delete_file
        Description :   This method delete the file from s3 bucket

        Output      :   The file is deleted from s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_file.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            s3_resource = self.get_s3_resource(table_name=table_name)

            s3_resource.objectect(bucket_name, file_name).delete()

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Deleted {file_name} from bucket_name {bucket_name}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def move_data(
        self, from_file_name, from_bucket_name, to_file_name, to_bucket_name, table_name
    ):
        """
        Method Name :   move_data
        Description :   This method moves the data from one bucket to other bucket

        Output      :   The data is moved from one bucket to another
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.move_data.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            self.copy_data(
                from_bucket_name=from_bucket_name,
                from_file_name=from_file_name,
                to_bucket_name=to_bucket_name,
                to_file_name=to_file_name,
                table_name=table_name,
            )

            self.delete_file(
                bucket_name=from_bucket_name,
                file=from_file_name,
                table_name=table_name,
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Moved {from_file_name} from bucket_name {from_bucket_name} to {to_bucket_name}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_files_from_folder(self, folder_name, bucket_name, table_name):
        """
        Method Name :   get_files_from_folder
        Description :   This method gets the files a folder in s3 bucket

        Output      :   A list of files is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_files_from_folder.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            lst = self.get_file_objectect(
                bucket_name=bucket_name, table_name=table_name, file_name=folder_name,
            )

            list_of_files = [objectect.key for objectect in lst]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got list of files from bucket_name {bucket_name}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return list_of_files

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def get_file_objectect(self, file_name, bucket_name, table_name):
        """
        Method Name :   get_file_objectect
        Description :   This method gets the file objectect from s3 bucket

        Output      :   A file objectect is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_file_objectect.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            bucket = self.get_bucket(bucket_name=bucket_name, table_name=table_name,)

            lst_objects = [
                objectect for objectect in bucket.objectects.filter(Prefix=file_name)
            ]

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Got {file_name} from bucket_name {bucket_name}",
            )

            func = lambda x: x[0] if len(x) == 1 else x

            file_objects = func(lst_objects)

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return file_objects

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def load_model(self, model_name, bucket_name, table_name, model_dir=None):
        """
        Method Name :   load_model
        Description :   This method loads the model from s3 bucket

        Output      :   A pandas series objectect consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.load_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            func = (
                lambda: model_name + self.file_format
                if model_dir is None
                else model_dir + model_name + self.file_format
            )

            model_file = func()

            self.log_writer.log(
                table_name == table_name, log_info=f"Got {model_file} as model file",
            )

            f_object = self.get_file_objectect(
                file_name=model_name, bucket_name=bucket_name, table_name=table_name
            )

            model_object = self.read_objectect(
                objectect=f_object, table_name=table_name, decode=False
            )

            model = pickle.loads(model_object)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Loaded {model_name} from bucket_name {bucket_name}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

            return model

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def save_model(self, model, model_dir, model_bucket_name, table_name, idx=None):
        """
        Method Name :   save_model
        Description :   This method saves the model into particular model directory in s3 bucket with kwargs

        Output      :   A pandas series objectect consisting of runs for the particular experiment id
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.save_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            model_name = self.model_utils.get_model_name(
                model=model, table_name=table_name
            )

            func = (
                lambda: model_name + self.file_format
                if model_name == "KMeans"
                else model_name + str(idx) + self.file_format
            )

            model_file = func()

            with open(file=model_file, mode="wb") as f:
                pickle.dump(model, f)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Saved {model_name} model as {model_file} name",
            )

            bucket_model_path = model_dir + "/" + model_file

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploading {model_file} to {model_bucket_name} bucket",
            )

            self.upload_file(
                from_file_name=model_file,
                to_file_name=bucket_model_path,
                bucket_name=model_bucket_name,
                table_name=table_name,
            )

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Uploaded  {model_file} to {model_bucket_name} bucket_name",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.log(
                table_name=table_name,
                log_message=f"Model file {model_name} could not be saved",
            )

            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

    def upload_df_as_csv(
        self, data_frame, local_file_name, bucket_file_name, bucket_name, table_name
    ):
        """
        Method Name :   upload_df_as_csv
        Description :   This method uploades a dataframe as csv file to s3 bucket

        Output      :   A dataframe is uploaded as csv file to s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.upload_df_as_csv.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            table_name=table_name,
        )

        try:
            data_frame.to_csv(local_file_name, index=None, header=True)

            self.log_writer.log(
                table_name=table_name,
                log_message=f"Created a local copy of dataframe with name {local_file_name}",
            )

            self.upload_file(
                from_file_name=local_file_name,
                to_file_name=bucket_file_name,
                bucket_name=bucket_name,
                table_name=table_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                table_name=table_name,
            )
