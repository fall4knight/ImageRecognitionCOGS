import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_q1_data():
    file_path = 'q1_data.txt'
    data = np.genfromtxt(file_path,dtype="f8,f8,f8,f8,S20",
    delimiter=',',names=['x1','x2','x3','x4','class'])

    train_data = np.concatenate((data[15:50],data[65:]))
    test_data = np.concatenate((data[:15],data[50:65]))
    X_train = np.vstack([np.array((1,x[0],x[1],x[2],x[3]))
                         for x in train_data])
    X_test = np.vstack([np.array((1,x[0],x[1],x[2],x[3])) for x in test_data])
    num_train = len(X_train)
    num_test = len(X_test)
    le = LabelEncoder()
    le.fit(data['class'])
    y_train = le.transform(train_data['class']).reshape(num_train,1)
    y_test = le.transform(test_data['class']).reshape(num_test,1)
    return X_train, X_test, y_train, y_test, le
