import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

look_back = 30

#載入模型
from keras.models import load_model
model = load_model('AI期末/filename.h5')

def create_dataset(ds, look_back=1):
    X_data, Y_data = [],[]

    for i in range(len(ds)-look_back):
        X_data.append(ds[i:(i+look_back), 0])
        Y_data.append(ds[i+look_back, 0])

    return np.array(X_data), np.array(Y_data)

df_test = pd.read_csv('AI期末/test6-8.csv')
X_test_set = df_test.iloc[:,8:9].values
X_test, Y_test = create_dataset(X_test_set, look_back)

pred_date = 7
for loon in range(pred_date):
    X_test_set = np.append(X_test_set, [[0]], axis=0)
    print(X_test_set)
    X_test, Y_test = create_dataset(X_test_set, look_back)
    print(len(X_test)-1)
    print(X_test[len(X_test)-1])
    pred = model.predict(X_test[len(X_test)-1])
    X_test_set[len(X_test_set)-1] = [pred]
    print(X_test_set)
