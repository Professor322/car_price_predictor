from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import re

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

    def to_dict(self):
        return {
            'name': self.name,
            'year': self.year, 
            'km_driven': self.km_driven,
            'fuel': self.fuel,
            'seller_type': self.seller_type,
            'transmission': self.transmission,
            'owner': self.owner,
            'mileage': self.mileage,
            'engine' : self.engine,
            'max_power': self.max_power,
            'torque' : self.torque,
            'seats' : self.seats
            }


class Items(BaseModel):
    objects: List[Item]


class CarPriceInferenceRunner():
    '''
    This class implements naive approach for running inference to predict car prices
    '''
    def __init__(self, prediction_model_params: str, std_scaler_file: str, ohe_file: str):
        '''
        @prediction_model_params - filename for the pickle object for the dict with ['coef'] and ['intercept'] keys
        @std_scaler_file - filename for the pickle object for the fitted StandardScaler
        @ohe_file - filename for the pickle object for the fitted OneHotEncoder
        '''
        with open(prediction_model_params, 'rb') as model_coef_file:
            model_params =  pickle.load(model_coef_file)
            self.model_coef = model_params['coef']
            self.model_intercept = model_params['intercept']

        with open(std_scaler_file, 'rb') as scaler_file:
            self.std_scaler = pickle.load(scaler_file)

        with open(ohe_file, 'rb') as ohe_file:
            self.ohe = pickle.load(ohe_file)
    
    def __preprocess_items(self, to_preprocess: Item or List[Item]):
        '''
        @brief
        > creates data frame for the given Item of given list of Items
        > gets rid of torque and name
        > convert mileage, engine and max power to numeric
        > converts seats to object type
        @to_preprocess - object of type Item or list of Items
        '''
        self.X = pd.DataFrame(columns=['name',
                                       'year',
                                       'selling_price',
                                       'km_driven',
                                       'fuel',
                                       'seller_type',
                                       'transmission',
                                       'owner',
                                       'mileage',
                                       'engine',
                                       'max_power',
                                       'torque',
                                       'seats'])
        # add items to pd.DataFrame then run preprocessing
        self.Items = to_preprocess
        if isinstance(to_preprocess, list):
            for item in to_preprocess:
                self.X.loc[len(self.X)] = item.to_dict()
        else:
            self.X.loc[len(self.X)] = to_preprocess.to_dict()
        
        self.X.drop(inplace=True, columns=['torque', 'name', 'selling_price'])
        self.X['mileage'] = pd.to_numeric(self.X['mileage'].map(lambda obj: re.sub('[^0-9.]', '', obj), na_action='ignore'),
                                  errors='coerce')
                                
        self.X['engine'] = pd.to_numeric(self.X['engine'].map(lambda obj: re.sub('[^0-9.]', '', obj), na_action='ignore'),
                                  errors='coerce')
        self.X['max_power'] = pd.to_numeric(self.X['max_power'].map(lambda obj: re.sub('[^0-9.]', '', obj), na_action='ignore'),
                                  errors='coerce')
        
        self.X['seats'] = self.X['seats'].astype(object)

    def __standardize(self):
        numeric_columns =  self.X.select_dtypes(include='number').columns
        self.X[numeric_columns] = self.std_scaler.transform(self.X[numeric_columns])

    def __encode(self) -> pd.DataFrame:
        cat_columns = self.X.select_dtypes(object).columns
        self.X[self.ohe.get_feature_names_out()] = self.ohe.transform(self.X[cat_columns])
        self.X.drop(columns=cat_columns, inplace=True)

    def fit(self, to_fit: Item or List[Item]) -> None:
        self.__preprocess_items(to_fit)
        self.__encode()
        self.__standardize()

    def predict(self) -> Item or List[Item]:
        self.Y = self.model_intercept + self.X @ self.model_coef
        return self.Y

    def fit_predict(self, to_fit: Item or List[Item]) -> Item or List[Item]:
        self.fit(to_fit)
        return self.predict()


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    inference_runner.fit(item)
    return inference_runner.predict()[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    inference_runner.fit(items)
    return list(inference_runner.predict())



inference_runner = CarPriceInferenceRunner(prediction_model_params='prediction_model_coef.pkl',
                                           std_scaler_file='scaler.pkl',
                                           ohe_file='encoder.pkl')


'''
Tests:

example_item1 = Item(name='Hyundai Grand i10 Sport', 
                    year=2017, 
                    selling_price=450000, 
                    km_driven=35000,
                    fuel='Petrol',
                    seller_type='Individual',
                    transmission='Manual',
                    owner='First Owner',
                    mileage='18.9 kmpl',
                    engine='1197 CC',
                    max_power='82 bhp',
                    torque='114Nm@ 4000rpm',
                    seats=5.0)

example_item2 = Item(name='Renault KWID RXT', 
                    year=2016, 
                    selling_price=330000, 
                    km_driven=20000,
                    fuel='Petrol',
                    seller_type='Individual',
                    transmission='Manual',
                    owner='First Owner',
                    mileage='25.17 kmpl',
                    engine='799 CC',
                    max_power='53.3 bhp',
                    torque='72Nm@ 4386rpm',
                    seats=5.0)

inference_runner = CarPriceInferenceRunner(prediction_model_coef_file='prediction_model_coef.pkl',
                                           std_scaler_file='scaler.pkl',
                                           ohe_file='encoder.pkl')

inference_runner.fit(example_item2)
price = inference_runner.predict()
print(f'price: {price[0]}$')

inference_runner.fit([example_item1, example_item2])
price = inference_runner.predict()
print(f'price: {price}')
'''