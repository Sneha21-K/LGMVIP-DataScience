import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# - # Load Data




data = pd.read_csv('NSE-TATAGLOBAL11.csv')




data.head()





data.tail()


# - # Sort with date




data['Date']= pd.to_datetime(data['Date'])
print(type(data.Date[0]))





df = data.sort_values(by='Date')
df.head()





df.reset_index(inplace= True)





df.head()





df_close = df['Close']





plt.plot(df_close)
plt.xlabel('No. of datapoint')
plt.ylabel('Close')
plt.title('Stock Prediction')





df1=df['Close']


# - # Prepare data




scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df_close).reshape(-1,1))
df1


# - # Splitting dataset into train and test




training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]





training_size,test_size


# - # convert an array of values into a dataset matrix




def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# - # reshape into X=t,t+1,t+2,t+3 and Y=t+4




time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)





print(X_train.shape), print(y_train.shape)





print(X_test.shape), print(ytest.shape)


# - # reshape input to be [samples, time steps, features] which is required for LSTM




X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# - # Create the Stacked LSTM model




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM





model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.summary()





model.fit(X_train,y_train,validation_split=0.1,epochs=10,batch_size=64,verbose=1)


# - # Lets Do the prediction and check performance metrics




train_predict=model.predict(X_train)
test_predict=model.predict(X_test)





import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train, train_predict))





from sklearn import metrics
np.round(metrics.r2_score(y_train, train_predict),2)


# - # calculate RMSE performance metrics




math.sqrt(mean_squared_error(ytest, test_predict))





np.round(metrics.r2_score(ytest, test_predict),2)


# - # Transformback to original form




train_predict =scaler.inverse_transform(train_predict)
test_predict =scaler.inverse_transform(test_predict)
y_train =scaler.inverse_transform(y_train.reshape(-1,1))
ytest =scaler.inverse_transform(ytest.reshape(1,-1))


# - # Plotting




#Shift train prediction for plotting
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#Shift test prediction for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2)+1:len(df1) - 1, :] = test_predict

#Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()





x_input = test_data[511:].reshape(1,-1)
x_input.shape





temp_input = list(x_input)
temp_input = temp_input[0].tolist()




from numpy import array
lst_output=[]
n_steps=100
nextNumberOfDays = 30
i=0

while(i<nextNumberOfDays):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)





day_new = np.arange(1,101)
day_pred = np.arange(101,161)





df3 = df1.tolist()
df3.extend(lst_output)





len(df1)





plt.plot(day_new, scaler.inverse_transform(df1[1935:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.xlabel('No. of datapoint')
plt.ylabel('Close')
plt.title('Stock Prediction')

