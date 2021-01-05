import numpy as np
import lightgbm as gbm
import xgboost as xgb
from tensorflow import keras

class XGBM:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

    def data_process(self):
        Length = []
        for i in range(len(self.Y)):
            self.X_train.append(self.X[i])
            self.Y_train.append(self.Y[i])
            Length.append(len(list(self.X[i])))

    def model(self):
        self.data_process()
        Model = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=50, objective='reg:squarederror')
        model = Model.fit(self.X_train,self.Y_train)#
        return model

class FNN:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

    def data_process(self):
        for i in range(len(self.Y)):
            self.X_train.append(self.X[i])
            self.Y_train.append(self.Y[i])

    def build_model(self,input_dim):
        multiplier_each = 6
        Num_Layer = 5
        layers_each = 5
        Model = keras.Sequential()
        Model.add(keras.layers.Dense(input_dim * multiplier_each, input_shape=(input_dim,)))

        for i in range(Num_Layer):
            Model.add(
                keras.layers.Dense(input_dim * multiplier_each * (layers_each - i) / layers_each, activation='relu'))
        Model.add(keras.layers.Dense(1, activation='sigmoid'))

        Model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse']
        )
        return Model

    def model(self):
        self.data_process()
        Input_Dim = len(self.X_train[0])
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        Model = self.build_model(Input_Dim)
        Model.fit(self.X_train,self.Y_train,epochs=1)
        return Model
