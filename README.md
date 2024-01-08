# Stock_Price_Prediction
I have build this stock Price Prediction model which was based on usage of Stacked LSTM which is just a fancy way saying that more 2 LSTM model have been used here stack by stack.
What has been in this project is since we all know that the time-series data are correlated to each other especially in case of the Stock Price Current price depend on the previous values
So for that I have not applied direct test train split but instead I have taken some initial contiguous portion as train and rest portion as test. Then I have normalize all with the help 
of the MinMax Scaler which brings down all the values between 0 and 1 this is necessary step cause LSTM are highly sensitive to the data values. Next up we have created dataset out of
all the values that we have that we have created following the rule that current value depend on the previous values so for each values 

where the dataset is like 0.2 0.3 0.4 0.1 0.7 ====> this we have segmented into the way given below so we can maintain the fact that each price of stock is depended on the previous values

f1     f2     f3     Y
0.2    0.4    0.3    0.1
0.4    0.3    0.1    0.7


This way I can have applied it to both trian and test dataset then I have given it to the LSTM layers to train for certain epochs.

Rest I have plot the graph for the prediction made !!!
![image](https://github.com/Siddharth133/Stock_Price_Prediction/assets/99598353/ce90d3ea-9d82-4a1d-85c7-f9d0957a3cbd)

POINTS TO BE REMEBERED :

1> While Working with TIME-SERIES data always make sure you take into consideration the relationship between each values.
2> LSTM are highly sensitive to the data values so whenever LSTM are used make sure you normalize all data.
3> For TIME-SERIES data always pick up the model which can retain the data for longer period as values are interdependent.
