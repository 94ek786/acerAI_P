import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt 

waku = []

test_amount = 10
for loop39 in range(test_amount):
  np.random.seed(loop39)
  df_train = pd.read_csv('AI期末/train1.csv', index_col='date',parse_dates=True)


  X_train_set = df_train.iloc[:,7:8].values

  sc = MinMaxScaler()
  X_train_set = sc.fit_transform(X_train_set)

  def create_dataset(ds, look_back=1):
    X_data, Y_data = [],[]
    for i in range(len(ds)-look_back):
      X_data.append(ds[i:(i+look_back), 0])
      Y_data.append(ds[i+look_back, 0])
    
    return np.array(X_data), np.array(Y_data)
  days = 5
  n_steps = days #輸入張量的維度數 
  n_features = 5 #輸入張量的維度
  look_back = 30
  X_train, Y_train = create_dataset(X_train_set, look_back)

  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))




  model = Sequential()
  model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units = 50, return_sequences = True,input_shape=(n_steps, n_features)))
  model.add(Dropout(0.05))
  model.add(LSTM(units = 50))
  model.add(Dense(1))
  model.summary()
  model.compile(loss='mse', optimizer='adam')

  model.fit(X_train, Y_train, epochs=100, batch_size=64)


  df_test = pd.read_csv('AI期末/test1.csv')
  X_test_set = df_test.iloc[:,8:9].values
  X_test, Y_test = create_dataset(X_test_set, look_back)
  old_shape = X_test.shape
  X_test = sc.transform(X_test.reshape(-1,1))
  X_test = np.reshape(X_test, (old_shape[0], old_shape[1],1))
  X_test_pred = model.predict(X_test)

  X_test_pred_price = sc.inverse_transform(X_test_pred)


  plt.plot(Y_test, color='red', label='Real Stock Price')
  plt.plot(X_test_pred_price, color='blue', label='Predicted Stock Price')
  plt.title('No.' + str(loop39+1) + ' 2022 Acer Stock Price')
  plt.xlabel('Day')
  plt.ylabel('Price')
  plt.legend()
  plt.savefig('img/' + str(loop39) + '-1.png')
  plt.clf()



#預測未來
  df_test = pd.read_csv('AI期末/test6-8.csv')
  X_test_set = df_test.iloc[:,8:9].values

  pred_date = 6
  for loon in range(pred_date):
      X_test_set = np.append(X_test_set, [[0]], axis=0)
      X_test, Y_test = create_dataset(X_test_set, look_back)
      old_shape = X_test.shape
      X_test = sc.transform(X_test.reshape(-1,1))
      X_test = np.reshape(X_test, (old_shape[0], old_shape[1],1))
      pred = model.predict(np.array([X_test[len(X_test)-1],]))
      pred = sc.inverse_transform(pred)
      #print(pred)
      X_test_set[len(X_test_set)-1] = [pred]
      
  #print(X_test_set)
  X_test, Y_test = create_dataset(X_test_set, look_back)
  plt.plot(Y_test, color='red')
  plt.title('No.' + str(loop39+1) + ' predicted data after 5/30')
  plt.xlabel('Day')
  plt.ylabel('Price')
  plt.legend()
  plt.savefig('img/' + str(loop39) + '-2.png')
  plt.clf()

  waku.append('No.' + str(loop39+1) + ' 6/8 price is ' + str(X_test_set[len(X_test_set)-1][0]))

for loop77 in waku:
  print(loop77)
