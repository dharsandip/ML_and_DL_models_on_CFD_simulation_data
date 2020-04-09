# This dataset is from the CFD simulation of simple 3D pipe flow case with heat transfer having one inlet, outlet and wall.  At inlet, we are varying the inlet velocity, temperature, turbulence intensity etc. and after solving the case we are reporting the total heat transfer rate or heat flux at outlet. For generating CFD simulation data, ANSYS-Fluent-ROM (Student version) was used under Workbench framework. In 3D ROM (Reduced Order Model), we created 100 design of experiments or design points and all of our input parameters (inlet velocity, inlet temperature, inlet turbulence intensity)  were varied for a specified range and case for each design point was solved in ANSYS-Fluent and we got the value of output parameter (total heat transfer rate or heat flux at outlet). Finally we had the data of 100 design points in a table where first 3 columes were the data for input parameters (independent variables) and the last colume was the data for output parameter (dependent variables). 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('pipe-flow-Fluent-ROM-data.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

y_test = y_test.reshape(-1, 1)
y_test = sc_y.transform(y_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 3))

# Adding the 2nd hidden layer
regressor.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))


# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 1, nb_epoch = 600)

y_pred = regressor.predict(X_test)


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3))
    regressor.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 1, epochs = 600)
errors = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = errors.mean()
variance = errors.std()


MSE_of_testset = np.mean((y_pred - y_test)**2)


y_pred = sc_y.inverse_transform(y_pred)
X_test = sc_X.inverse_transform(X_test)
y_test = sc_y.inverse_transform(y_test)

X_train = sc_X.inverse_transform(X_train)
y_train = sc_y.inverse_transform(y_train)


plt.scatter(X_test[:, 0], y_test, s=100, c = 'red', label = 'CFD simulation data')
plt.scatter(X_test[:, 0], y_pred, s=100, c = 'blue', label = 'ANN predictions')
plt.xlabel('Inlet Velocity')
plt.ylabel('Total Heat Transfer Rate at outlet')
plt.title('Comparison of ANN predictions with original CFD simulation data for the testset')
plt.legend()
plt.show()


plt.scatter(X_test[:, 1], y_test, s=100, c = 'red', label = 'CFD simulation data')
plt.scatter(X_test[:, 1], y_pred, s=100, c = 'blue', label = 'ANN predictions')
plt.xlabel('Inlet Temperature')
plt.ylabel('Total Heat Transfer Rate at outlet')
plt.title('Comparison of ANN predictions with original CFD simulation data for the testset')
plt.legend()
plt.show()

