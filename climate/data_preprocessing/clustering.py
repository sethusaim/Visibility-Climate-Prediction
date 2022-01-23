import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from src.file_operations.file_methods import File_Operation
from utils.logger import App_Logger
from utils.main_utils import read_params


class KMeansClustering:
    """
    This class shall  be used to divide the data into clusters before training.

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None

    """

    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.db_name = db_name

        self.collection_name = collection_name

        self.log_writter = App_Logger()

    def elbow_plot(self, data):
        """
        Method Name: elbow_plot
        Description: This method saves the plot to decide the optimum number of clusters to the file.
        Output: A picture saved to the directory
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the elbow_plot method of the KMeansClustering class",
        )

        wcss = []

        try:
            for i in range(1, self.config["kmeans_cluster"]["max_clusters"]):
                kmeans = KMeans(
                    n_clusters=i,
                    init=self.config["kmeans_cluster"]["init"],
                    random_state=self.config["base"]["random_state"],
                )

                kmeans.fit(data)

                wcss.append(kmeans.inertia_)

            plt.plot(range(1, self.config["kmeans_cluster"]["max_clusters"]))

            plt.title("The Elbow Method")

            plt.xlabel("Number of clusters")

            plt.ylabel("WCSS")

            plt.savefig(self.config["elbow_plot_fig"])

            self.kn = KneeLocator(
                x=range(1, self.config["kmeans_cluster"]["max_clusters"]),
                y=wcss,
                curve=self.config["kmeans_cluster"]["knee_locator"]["curve"],
                direction=self.config["kmeans_cluster"]["knee_locator"]["direction"],
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="The optimum number of clusters is: "
                + str(self.kn.knee)
                + " . Exited the elbow_plot method of the KMeansClustering class",
            )

            return self.kn.knee

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : KMeansClustering, Method : elbow_plot, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding the number of clusters failed. \
                    Exited the elbow_plot method of the KMeansClustering class",
            )

            raise Exception(
                f"Exception occured in Class : KMeansClustering, Method : elbow_plot, Error : {str(e)}"
            )

    def create_clusters(self, data, number_of_clusters):
        """
        Method Name :   create_clusters
        Description :   Create a new dataframe consisting of the cluster information.
        Output      :   A datframe with cluster column
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.0
        Revisions   :   None

        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the create_clusters method of the KMeansClustering class",
        )

        self.data = data

        try:
            self.kmeans = KMeans(
                n_clusters=number_of_clusters,
                init=self.config["kmeans_clluster"]["init"],
                random_state=self.config["base"]["random_state"],
            )

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.file_op = File_Operation(self.db_name, self.collection_name)

            self.save_model = self.file_op.save_model(
                self.kmeans, self.config["model_names"]["kmeans_model_name"]
            )

            self.data["Cluster"] = self.y_kmeans

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="succesfully created "
                + str(self.kn.knee)
                + "clusters. Exited the create_clusters method of the KMeansClustering class",
            )

            return self.data, self.kmeans

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Exception occured in Class : KMeansClustering, Method : create_clusters, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Fitting the data to clusters failed. \
                    Exited the create_clusters method of the KMeansClustering class",
            )

            raise Exception(
                f"Exception occured in Class : KMeansClustering, Method : create_clusters, Error : {str(e)}"
            )
