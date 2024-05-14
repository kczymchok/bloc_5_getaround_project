# import uvicorn
# import pandas as pd 
# from pydantic import BaseModel
# from fastapi import FastAPI, Request
# import numpy as np
# import joblib
# from typing import List
# import json

# description = """
# # How many peanuts per day cost my car ? #

# ## Overview ##

# This is an API allowing the user to predict the rental price per day in euro for a car for which
# he selected criteria (color, fuel, model of the car etc). The user can put different options of car
# criteria. The url to use to access the predict endpoint is the following:
# https://getaround-api-elo-16ab161d9781.herokuapp.com/predict 

# ## Input data recommandations ##
# The car criteria should be a JSON, written exactly as in the example following (for 2 options here):

# {
#   "car_options": [
#     {
#       "model_key": "Citroën",
#       "mileage": 140411,
#       "engine_power": 100,
#       "fuel": "diesel",
#       "paint_color": "black",
#       "car_type": "convertible",
#       "private_parking_available": true,
#       "has_gps": true,
#       "has_air_conditioning": false,
#       "automatic_car": false,
#       "has_getaround_connect": true,
#       "has_speed_regulator": true,
#       "winter_tires": true
#     },
#     {
#       "model_key": "Citroën",
#       "mileage": 13929,
#       "engine_power": 317,
#       "fuel": "petrol",
#       "paint_color": "grey",
#       "car_type": "convertible",
#       "private_parking_available": true,
#       "has_gps": true,
#       "has_air_conditioning": false,
#       "automatic_car": false,
#       "has_getaround_connect": false,
#       "has_speed_regulator": true,
#       "winter_tires": true
#     }
#   ]
# }

# ## Output data ##

# Here is the answer that you would receive for the example above:

# {
#   "predictions": [
#     "Option 1: 88 €",
#     "Option 2: 156 €"
#   ]
# }

# ## Get your predictions ## 

# You can get your predictions via python:

# import requests\n
# response = requests.post("https://getaround-api-elo-f1df83a28de6.herokuapp.com/predict", json={'car_options': [{...}, {...}]})
# print(response.json())

# """

# app = FastAPI(
#     title="GetAround API",
#     description=description,
#     version="0.1",
#     contact={"name": "Kevin Chok"},
# )

# # We want to be able to write a request ourselves. For this, we need to create a class with pydantic 
# # and Base Model that contains all our explanative variables. In the example of the description above,
# # the user makes 2 possible options of car criteria. So we need to define also a class for the list 
# # containing the different options of car criteria.

# class CarCriteria(BaseModel):
#     model_key: str
#     mileage: int
#     engine_power: int
#     fuel: str
#     paint_color: str
#     car_type: str
#     private_parking_available: bool
#     has_gps: bool
#     has_air_conditioning: bool
#     automatic_car: bool
#     has_getaround_connect: bool
#     has_speed_regulator: bool
#     winter_tires: bool

# class CarOptions(BaseModel):
#     car_options: List[CarCriteria]

# # Then we need to define several functions. One is loading the model and other parameters needed, 
# # one is transforming the input_data with the same preprocessing as used when creating the model, 
# # and the last one is doing the prediction.

# def load_model():
#     model_file = joblib.load('CatBoost_model.joblib')
#     model = model_file['model']
#     feature_encoder = model_file['feature_encoder']
#     scaler = model_file['scaler']   
#     return model, feature_encoder, scaler

# def preprocess_data(input_data,feature_encoder):
#     # Create the list that will contain the preprocessed options:
#     preprocessed_data = []
#     for option in input_data:
#         # Put each option of criteria into a dataframe
#         option_df = pd.DataFrame([option.dict()])
#         # Preprocess this dataframe
#         preprocessed_option_df = feature_encoder.transform(option_df)
#         preprocessed_option_array = preprocessed_option_df.toarray()
#         # Add preprocessed option to the final list:
#         preprocessed_data.append(preprocessed_option_array)
#     # Convert the list of preprocessed data to a numpy array
#     preprocessed_data = np.concatenate(preprocessed_data, axis=0)
#     return preprocessed_data


# def predict_data(model, preprocessed_data, scaler):
#     normalized_predictions = model.predict(preprocessed_data)
#     # Reshape the normalized predictions if needed
#     if len(normalized_predictions.shape) == 1:
#         normalized_predictions = normalized_predictions.reshape(-1, 1)   
#     # Inverse transform the predictions to the original scale
#     predictions = scaler.inverse_transform(normalized_predictions)
#     return predictions.tolist()


# # Then we define the endpoints of our FastAPI. Here we used only one endpoint which is a POST 
# # method since we need an input from the user concerning the car criteria. And this endpoint will
# # make the prediction and give back an answer in JSON.


# @app.post("/predict", tags=["Machine Learning"])
# async def predict(car_options: CarOptions):
#     model, feature_encoder, scaler = load_model()
#     preprocessed_data = preprocess_data(car_options.car_options, feature_encoder)
#     predictions = predict_data(model, preprocessed_data, scaler)
#     formatted_predictions = [f"Option {i+1}: {round(pred[0])} €" for i, pred in enumerate(predictions)]
#     return {"predictions": formatted_predictions}


# if __name__=="__main__":
#     uvicorn.run(app, host="0.0.0.0", port=4000) # Here you define your web server to run the `app` variable (which contains FastAPI instance), with a specific host IP (0.0.0.0) and port (4000)


import gc
import mlflow
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

description = """
Welcome to my rental price predictor API !\n
Submit the characteristics of your car and a Machine Learning model, trained on GetAround data, will recommend you a price per day for your rental. 

**Use the endpoint `/predict` to estimate the daily rental price of your car !**
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Use this endpoint for getting predictions"
    }
]

app = FastAPI(
    title="💸 Car Rental Price Predictor",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

class Car(BaseModel):
    model_key: Literal['Citroën','Peugeot','PGO','Renault','Audi','BMW','Mercedes','Opel','Volkswagen','Ferrari','Mitsubishi','Nissan','SEAT','Subaru','Toyota','other'] 
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: Literal['diesel','petrol','other']
    paint_color: Literal['black','grey','white','red','silver','blue','beige','brown','other']
    car_type: Literal['convertible','coupe','estate','hatchback','sedan','subcompact','suv','van']
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Redirect automatically to /docs (without showing this endpoint in /docs)
@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

# Make predictions
@app.post("/predict", tags=["Predictions"])
async def predict(cars: List[Car]):
    # clean unused memory
    gc.collect(generation=2)

    # Read input data
    car_features = pd.DataFrame(jsonable_encoder(cars))

    # Load model as a PyFuncModel.
    logged_model = 'runs:/c7971fb7c3c446b28547044f7a1638bb/car_rental_price_predictor'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict and format response
    prediction = loaded_model.predict(car_features)
    response = {"prediction": prediction.tolist()}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)

