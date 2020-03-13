import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sklearn.tree

from sklearn.model_selection import train_test_split


# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

#data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
#predictors.remove('Target')
predictors.remove('Unnamed: 0')
X = np.array(data)[:,1:]


x_train, x_test, y_train, y_test = train_test_split(X, purchasebin, train_size=0.5)


pesos = []
test_desv = []
train_desv = []
f1_test = []
f1_train = []

for i in range(1,11):
    f1_train_i = []
    f1_test_i = []
    pesos_i = np.zeros(14)
    arbol = sklearn.tree.DecisionTreeClassifier(max_depth=i)
    
    for j in range(100):
        indices = np.random.choice(np.arange(len(y_train)), len(y_train))
    
        x_train_new = x_train[indices,:]
        y_train_new = y_train[indices]
    
        arbol.fit(x_train_new, y_train_new)
        pesos_i += arbol.feature_importances_
        f1_train_i.append(sklearn.metrics.f1_score(y_train_new,arbol.predict(x_train_new)))
        f1_test_i.append(sklearn.metrics.f1_score(y_test,arbol.predict(x_test)))
        
    f1_test.append(np.mean(f1_test_i))
    f1_train.append(np.mean(f1_train_i))
    pesos.append(pesos_i/100)
    test_desv.append(np.std(f1_test_i))
    train_desv.append(np.std(f1_train_i))
    
pesos = np.array(pesos)


plt.figure()
plt.errorbar(range(1,11),f1_train,yerr=train_desv,fmt='o',label='Train')
plt.errorbar(range(1,11),f1_test,yerr=test_desv,fmt='o',label='Test')
plt.legend()
plt.ylabel('Average F1-Score')
plt.xlabel('Max Depth')
plt.savefig('F1_training_test.png')


plt.figure()
print(np.shape(predictors))
for i in range(0,14):
    plt.plot(range(1,11),pesos[:,i],label=predictors[i])
plt.legend()
plt.ylabel('Average feature importance')
plt.xlabel('Max Depth')
plt.savefig('features.png')