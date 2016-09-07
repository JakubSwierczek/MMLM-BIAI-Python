import  pandas as pd
import  numpy as np
import  csv
import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


train, test, submission = data.GetTrainAndTestAndSubmission()
target = train.result.values
test_target = test.result.values
test_team1 = test.team1.values
test_team2 = test.team2.values
# usuniecie kolumny z wynikiem 1
train.drop('result', inplace=True, axis=1)
test.drop('result', inplace=True, axis=1)
# puste miejsca na -1
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)
submissionids = submission.Id.values
submission.drop(['Id', 'Pred'], inplace=True, axis=1)
submission.fillna(1, inplace=True)
print('Trening...')

# tworzymy las losowy.
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
print('Testing...')
test_pred = extraTreesClasiifier.predict_proba(test)
test_pred = test_pred[:,1]
loss = log_loss(test_target, test_pred)
test_result = pd.DataFrame({
    'team1': test_team1,
    'team2': test_team2,
    'target': test_target,
    'pred': np.round(test_pred, 0).astype(int)
})
cols = ['team1', 'team2', 'target', 'pred']
test_result = test_result[cols]
test_result.to_csv("test.csv", index=False)

print(loss)
print('Predict...')
# Przewidywanie wyników dla zestawu testowego. 
submission_pred = extraTreesClasiifier.predict_proba(submission)

submission = pd.DataFrame({'Id': submissionids,
                            'Pred': submission_pred[:,1]})
submission.to_csv('submission.csv', index=False)
 