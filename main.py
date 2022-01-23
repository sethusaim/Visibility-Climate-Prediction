import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from climate.model.load_production_model import LoadProdModel
from climate.model.predictionFromModel import prediction
from climate.model.trainingModel import trainModel
from climate.validation_insertion.prediction_validation_insertion import pred_validation
from climate.validation_insertion.train_validation_insertion import train_validation
from utils.read_params import read_params

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()

config = read_params()

templates = Jinja2Templates(directory=config["templates"]["dir"])

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        config["templates"]["index_html_file"], {"request": request}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        path = config["data_source"]["train_data_source"]

        train_valObj = train_validation(path)

        train_valObj.train_validation()

        trainModelObj = trainModel()

        num_clusters = trainModelObj.trainingModel()

        loadProdModelObj = LoadProdModel(num_clusters)

        loadProdModelObj.load_production_model()

    except Exception as e:
        return "Error Occurred! %s" % e

    return "Training successfull!!"


@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        if request.json is not None:
            path = request.json["filepath"]

            pred_val = pred_validation(path)

            pred_val.prediction_validation()

            pred = prediction(path)

            path, json_predictions = pred.predictionFromModel()

            return Response(
                "Prediction File created at !!!"
                + str(path)
                + "and few of the predictions are "
                + str(json.loads(json_predictions))
            )

        elif request.form is not None:
            path = request.form["filepath"]

            pred_val = pred_validation(path)

            pred_val.prediction_validation()

            pred = prediction(path)

            path, json_predictions = pred.predictionFromModel()

            return Response(
                "Prediction File created at "
                + str(path)
                + " and few of the predictions are "
                + str(json.loads(json_predictions))
            )

        else:
            print("Nothing Matched")

    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    host = config["app"]["host"]

    port = config["app"]["port"]

    uvicorn.run(app, host=host, port=port)