import  pandas as pd
import  numpy as np
import  csv
import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


train, test = data.GetTrainAndTest()

target = train.result.values
# usuniecie kolumny z wynikiem 1
train.drop('result', inplace=True, axis=1)
# puste miejsca na -1
train.fillna(-1, inplace=True)

testids = test.Id.values
test.drop(['Id', 'Pred'], inplace=True, axis=1)
test.fillna(1, inplace=True)

standardScaler = StandardScaler()

# Zaokrąglenie danych do 4 miejsc po przecinku z dopasowanej tablicy
# fit_transform - obliczanie wartości oczekiwanej oraz obliczanie standaryzacji i dopasowanie do zbioru
train[train.columns] = np.round(standardScaler.fit_transform(train), 4)
# transform - dokonuje standaryzacji 
test[test.columns] = np.round(standardScaler.transform(test), 4)
print('Trening...')

# tworzymy drzewo decyzyjne
# n_estimators - liczba drzew w lesie
# max_features - liczba kolumn 
# min_samples_split - minimalna liczba wymagana do rozdzielenia wezla (domyslnie 2)
# min_samples_leaf - minimalna liczba wymagana do liscia (domyslnie 1)
# max_depth - maksymalna glebokosc drzewa (domyślnie None)
# n_jobs - ile watkow - jezeli -1 to liczba rdzeni
extraTreesClasiifier = ExtraTreesClassifier(n_estimators=100 ,max_features = 27,criterion= 'entropy',min_samples_split= 1,
                            max_depth=None, min_samples_leaf= 1, n_jobs = -1)
# dopasowanie drzewa.
extraTreesClasiifier.fit(train,target) 
# przwidywanie dla zbioru treningowego
train_pred = extraTreesClasiifier.predict_proba(train)

loss = log_loss(target, train_pred[:,1])
print(loss)

print('Predict...')

# Przewidywanie wyników dla zestawu testowego. 
test_pred = extraTreesClasiifier.predict_proba(test)

submission = pd.DataFrame({'Id': testids,
                            'Pred': test_pred[:,1]})
submission.to_csv('submission.csv', index=True)
 