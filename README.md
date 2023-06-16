# StockHive
StockHive: Smart Predictions for Busy Traders


This main.py file contains code for training an LSTM model to predict stock prices and making predictions for the next day. The code uses historical stock data from Yahoo Finance and implements the model using TensorFlow and Keras libraries.

## Prerequisites
Before running the code, make sure you have the following libraries installed:

numpy
matplotlib
pandas
pandas_datareader
datetime
scikit-learn
TensorFlow
You can install the required libraries using pip:

## Copy code
pip install numpy matplotlib pandas pandas_datareader scikit-learn tensorflow


##Usage
Set the company symbol and date range:

Modify the company variable to the desired stock symbol (e.g., 'FB' for Facebook).
Adjust the start and end variables to define the date range for the historical stock data.
Run the code:

Execute the main.py file using a Python interpreter.
The code will perform the following steps:

Fetch historical stock data:

The code retrieves the historical stock data for the specified company and date range from Yahoo Finance using the pandas_datareader library.


#Prepare the data:

The stock data is normalized using the MinMaxScaler from scikit-learn, which scales the data to a range of 0 to 1.
The normalized data is then split into input-output pairs, where each input consists of a sequence of "prediction_days" stock prices used to predict the next price.
The input data is reshaped to fit the LSTM model's input shape.
Build the LSTM model:

The code creates a sequential model using the Sequential class from Keras.
Three LSTM layers with dropout are added to the model, followed by a dense layer for predicting the next closing value.
The model is compiled with the Adam optimizer and the mean squared error loss function.
The model is trained on the training data using the specified number of epochs and batch size.
Test the model accuracy on existing data:

Test data for the specified company from the past year up to the current date is fetched.
The actual stock prices from the test data are retrieved for comparison with the predicted prices.
The historical and test data are concatenated to create a complete dataset for prediction.
The input data for the model prediction is obtained by taking the last "prediction_days" number of prices from the total dataset and normalizing them using the same scaler used for the training data.
Predictions are made on the test data using the trained model.
The predicted prices are inverse transformed from the scaled values to their original values.
A plot is generated using matplotlib to visualize the actual and predicted prices.
Predict the next day:

Input data for predicting the next day is prepared by taking the last "prediction_days" days from the model_inputs array.
The data is reshaped to match the model's input dimensions.
The model predicts the price for the next day.
The predicted price is inverse transformed from the scaled value to its original value.
The predicted price for the next day is printed.
Note: The code assumes that you have internet access to fetch the stock data from Yahoo Finance.





