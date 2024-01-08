
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = load_model('my_model.h5')

# Assuming 'df' is your DataFrame with stock prices
# You will need to load it from a file or a database
df = pd.read_csv('AAPL.csv')
df1 = df.reset_index()['close']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(df1).reshape(-1, 1))
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data, test_data = scaled_data[0:training_size], scaled_data[training_size:len(scaled_data)]
import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)

x_input=test_data[341:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()


def predict_next_10_days(temp_input, model):
    lst_output = []
    n_steps = 100
    i = 0
    while i < 30:
        # Convert to NumPy array and ensure it's 1D
        temp_input_np = np.array(temp_input).flatten()

        # Pad if necessary
        if len(temp_input_np) < n_steps:
            temp_input_np = np.pad(temp_input_np, (n_steps - len(temp_input_np), 0), 'constant', constant_values=(0))

        # Reshape for prediction
        x_input_array = temp_input_np[-n_steps:].reshape(1, n_steps, 1)

        # Predict
        yhat = model.predict(x_input_array, verbose=0)
        print(f"{i} day output {yhat}")

        # Update temp_input
        temp_input = np.append(temp_input_np, yhat[0])

        lst_output.append(yhat[0])
        i += 1

    return lst_output



# def predict_next_10_days(temp_input, model):
#     lst_output = []
#     n_steps = 100
#     i = 0
#     while i < 30:
#         if len(temp_input) > n_steps:
#             x_input = np.array(temp_input[1:])
#             x_input = x_input.reshape(1, n_steps, 1)
#             yhat = model.predict(x_input, verbose=0)
#             print(f"{i} day output {yhat}")
#             temp_input.extend(yhat[0].tolist())  # If temp_input is a list
#             temp_input = temp_input[1:]
#             lst_output.extend(yhat.tolist())
#             i += 1
#         else:
#             x_input = np.array(temp_input).reshape(1, n_steps, 1)
#             yhat = model.predict(x_input, verbose=0)
#             print(yhat[0])
#             temp_input.extend(yhat[0].tolist())  # If temp_input is a list
#             lst_output.extend(yhat.tolist())
#             i += 1
#     return lst_output


def plot_stock_prediction(historical_data, predicted_data):
    plt.figure(figsize=(10, 5))
    plt.plot(historical_data, label='Historical Data')

    # This assumes predicted_data is a list of predictions, and historical_data is a numpy array
    # Get the last time step from the historical data
    last_time_step = len(historical_data) - 1
    # Prepare the time steps for the predicted data
    predicted_time_steps = np.arange(last_time_step + 1, last_time_step + len(predicted_data) + 1)

    # Invert the predicted data if it was scaled
    predicted_data_inverted = scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1)).flatten()

    # Plot the historical and forecast data
    plt.plot(predicted_time_steps, predicted_data_inverted, label='Forecast', color='orange')

    plt.legend()
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


# def plot_stock_prediction(historical_data, predicted_data):
#     plt.figure(figsize=(10,5))
#     plt.plot(historical_data, label='Historical Data')
#     day_pred = np.arange(101, 131)
#     plt.plot(np.arange(len(historical_data), len(historical_data) + len(predicted_data)), predicted_data, label='Forecast', color='orange')
#     plt.plot(day_pred, scaler.inverse_transform(predicted_data))
#     plt.legend()
#     plt.title("Stock Price Prediction")
#     plt.xlabel("Time")
#     plt.ylabel("Price")
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#     buf.close()
#     return image_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Preprocess the last 100 days of data
    last_100_days = scaled_data[-100:]
    # Predict the next 10 days
    predicted_values = predict_next_10_days(last_100_days, model)
    # Plot the results
    plot_img_base64 = plot_stock_prediction(df1.values, predicted_values)
    # Return the image and the values
    # Convert each NumPy array in the list to a list
    predicted_values_list = [arr.tolist() for arr in predicted_values]

    # Return the image and the values
    return jsonify({'image': plot_img_base64, 'predicted_values': predicted_values_list})

    # return jsonify({'image': plot_img_base64, 'predicted_values': predicted_values.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
