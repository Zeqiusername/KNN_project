from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))
# returns a numpy array

predict = 'class'

X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
acc = knn.score(X_test,y_test)
print(acc)
# gives a value of 90.6%

predicted = knn.predict(X_test)
print(X_test[:10])
names = {'unacc','acc','good','vgood'}
for x in range(len(predicted)):
    if predicted[x] != y_test[x]:
        Correct = 0
    else:
        Correct = 1
    print('Predicted:', predicted[x], 'Data: ', X_test[x], 'Actual', y_test[x],"Correctness:",Correct)

