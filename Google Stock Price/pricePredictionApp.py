
# Part 1 - Data Önişleme


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Training seti import et
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set) #training_set_scaled  0-1 arasına getiriyor seti.

# 60 timestep ve 1 outputu olan data yapısını kur
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - RNN'i kur

# Importing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# RNN'i initialize et
regressor = Sequential()

# İlk LSTM'i ekle ve Dropout yap

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# İkinci LSTM'i ekle ve Dropout yap

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Üçüncü LSTM'i ekle ve Dropout yap

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Dördüncü LSTM'i ekle ve Dropout yap
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer'ı ekle
regressor.add(Dense(units = 1))

#  RNN'i compile et
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# RNN'i training sete fitting et
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Öngörüleri yapma ve sonucu görselleştirme

# 2017'deki gerçek stok fiyatını al
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# 2017'deki öngörülen stok fiyatını getir
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)




# Sonuçları görselleştir
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Modeli kaydet
from keras.models import model_from_json
from keras.models import load_model

  
regressor_json = regressor.to_json()
with open("001.json", "w") as json_file:
    json_file.write(regressor_json)
regressor.save_weights("001.h5")

##########################################################
## Kendi oluşturacağın ####################################
#dataset_test2 = pd.read_csv('Google_Stock_Price_Test2.csv')
# 
#dataset_total2 = pd.concat((dataset_train['Open'], dataset_test2['Open']), axis = 0)
#inputs2 = dataset_total2[len(dataset_total2) - len(dataset_test2) - 60:].values
#inputs2 = inputs2.reshape(-1,1)
#inputs2 = sc.transform(inputs2)
#
#
#X_testSingle = []
#
#for i in range(60, 80):
#    X_testSingle.append(inputs2[i-60:i, 0])
#X_testSingle = np.array(X_testSingle)
#X_testSingle = np.reshape(X_testSingle, (X_testSingle.shape[0], X_testSingle.shape[1], 1))
#predicted_stock_price2 = regressor.predict(X_testSingle)
#predicted_stock_price2 = sc.inverse_transform(predicted_stock_price2)
#
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price2, color = 'green', label = 'Predicted Google Stock Price')
#plt.title('Google Stock Price Prediction')
#plt.xlabel('Time')
#plt.ylabel('Google Stock Price')
#plt.legend()
#plt.show()

########################################################
