import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# set the company and date range for the stock data
company = 'FB'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# get stock data for the specified company and date range from Yahoo Finance
data = web.DataReader(company, 'yahoo', start, end)

#Prepare Data
# normalize the stock data using MinMaxScaler to put it in the range of 0 to 1
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Close'].values.reshape(-1,1))

# set the number of days to predict into the future
prediction_days = 60

# create the training data by splitting the normalized stock data into arrays of size "prediction_days"
# where each array represents a set of "prediction_days" stock prices used to predict the next price
# x_train will be an array of arrays, where each inner array has "prediction_days" elements and contains the input values
# y_train will be an array with the output values that correspond to the next stock price after each set of "prediction_days" stock prices
x_train =[]
y_train=[]

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to 3 dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# build the LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

'''Test the model accuracy on existing data'''

# load test data for the specified company from the past year up to the current date
test_start=dt.datetime(2020, 1, 1)
test_end=dt.datetime.now()
test_data = web.DataReader(company, 'yahoo', test_start, test_end)

# get the actual stock prices from the test data to compare with the predicted prices
actual_prices=test_data['Close'].values

# concatenate the historical and test data to get the complete dataset for prediction
total_dataset=pd.concat((data['Close'], test_data['Close']), axis=0)

# get the input data for the model prediction by taking the last "prediction_days" number of prices from the total dataset
# and normalizing them using the same scaler used for the training data
model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions on Test Data
x_test=[]

# create x_test list that holds the test data inputs for the model by iterating through the model_inputs 
# array starting from the index of the prediction_days up to the end of the model_inputs array. 
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

# Convert the x_test list to a numpy array and reshape it to the right dimensions for the model.
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the predicted prices from the trained model
predicted_prices=model.predict(x_test)

# Inverse transform the predicted prices from the scaled values to their original values
predicted_prices=scaler.inverse_transform(predicted_prices)

# Plot the test predictions using matplotlib
plt.plot(actual_prices, color = "black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

#Predict Next Day
# prepare input data for predicting the next day by getting the last prediction_days days from model_inputs array and
# storing them in real_data
real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]

# Convert real_data list to a numpy array and reshape it to the right dimensions for the model
real_data = np.array(real_data)
real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

# get the prediction for the next day from the trained model
prediction=model.predict(real_data)

# Inverse transform the predicted price from the scaled value to its original value
prediction = scaler.inverse_transform(prediction)

# print the predicted price for the next day
print(f"Prediction: {prediction}")