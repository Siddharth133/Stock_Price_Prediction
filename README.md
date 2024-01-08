# Stock Price Prediction

I have built a Stock Price Prediction model using Stacked LSTM, which means utilizing more than two LSTM models stacked together. This project leverages the fact that time-series data, such as stock prices, are correlated, with current prices dependent on previous values.

Rather than using a direct train-test split, I have partitioned the initial contiguous portion of the data for training and the remaining for testing. All data is normalized using the MinMax Scaler to scale values between 0 and 1, a crucial step as LSTMs are sensitive to data scale.

The dataset is prepared in a way that respects the dependence of each stock price on its preceding values. For instance, from a sequence like 0.2, 0.3, 0.4, 0.1, 0.7, it is transformed into a structured format where each value is predicted based on its predecessors:

| f1   | f2   | f3   | Y   |
|------|------|------|-----|
| 0.2  | 0.4  | 0.3  | 0.1 |
| 0.4  | 0.3  | 0.1  | 0.7 |

This approach is applied to both the training and testing datasets, and then the data is fed into the LSTM layers for training over several epochs.

Finally, a graph is plotted to visualize the predictions made by the model.
![image](https://github.com/Siddharth133/Stock_Price_Prediction/assets/99598353/ce90d3ea-9d82-4a1d-85c7-f9d0957a3cbd)

**Points to Remember:**
1. When working with time-series data, always consider the relationship between values.
2. Since LSTMs are sensitive to data values, ensure all data is normalized.
3. For time-series data, choose models that can retain information over longer periods due to the interdependence of values.
