import  pandas as pd
import  numpy as np
import  csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

def Aggregate(teamcompactresults1,
              teamcompactresults2,
              merged_results,
              regularseasoncompactresults):
    winningteam1compactresults = pd.merge(how='left',
                                          left=teamcompactresults1,
                                          right=regularseasoncompactresults,
                                          left_on=['year', 'team1'],
                                          right_on=['Season', 'Wteam'])
    winningteam1compactresults.drop(['Season',
                                     'Daynum',
                                     'Wteam',
                                     'Lteam',
                                     'Lscore',
                                     'Wloc',
                                     'Numot'],
                                    inplace=True,
                                    axis=1)
    grpwinningteam1resultsaverage =  \
        winningteam1compactresults.groupby(['year', 'team1']).mean()
    winningteam1resultsaverage = grpwinningteam1resultsaverage.reset_index()
    winningteam1resultsaverage.rename(columns={'Wscore': 'team1WAverage'},
                                      inplace=True)
    grpwinningteam1resultsmin =  \
        winningteam1compactresults.groupby(['year', 'team1']).min()
    winningteam1resultsmin = grpwinningteam1resultsmin.reset_index()
    winningteam1resultsmin.rename(columns={'Wscore': 'team1Wmin'},
                                  inplace=True)
    grpwinningteam1resultsmax =  \
        winningteam1compactresults.groupby(['year', 'team1']).max()
    winningteam1resultsmax = grpwinningteam1resultsmax.reset_index()
    winningteam1resultsmax.rename(columns={'Wscore': 'team1Wmax'},
                                  inplace=True)
    grpwinningteam1resultsmedian =  \
        winningteam1compactresults.groupby(['year', 'team1']).median()
    winningteam1resultsmedian = grpwinningteam1resultsmedian.reset_index()
    winningteam1resultsmedian.rename(columns={'Wscore': 'team1Wmedian'},
                                     inplace=True)
    grpwinningteam1resultsstd =  \
        winningteam1compactresults.groupby(['year', 'team1']).std()
    winningteam1resultsstd = grpwinningteam1resultsstd.reset_index()
    winningteam1resultsstd.rename(columns={'Wscore': 'team1Wstd'},
                                  inplace=True)
    losingteam1compactresults = pd.merge(how='left',
                                         left=teamcompactresults1,
                                         right=regularseasoncompactresults,
                                         left_on=['year', 'team1'],
                                         right_on=['Season', 'Lteam'])
    losingteam1compactresults.drop(['Season',
                                    'Daynum',
                                    'Wteam',
                                    'Lteam',
                                    'Wscore',
                                    'Wloc',
                                    'Numot'],
                                   inplace=True,
                                   axis=1)
    grplosingteam1resultsaverage = \
        losingteam1compactresults.groupby(['year', 'team1']).mean()
    losingteam1resultsaverage = grplosingteam1resultsaverage.reset_index()
    losingteam1resultsaverage.rename(columns={'Lscore': 'team1LAverage'},
                                     inplace=True)
    grplosingteam1resultsmin = \
        losingteam1compactresults.groupby(['year', 'team1']).min()
    losingteam1resultsmin = grplosingteam1resultsmin.reset_index()
    losingteam1resultsmin.rename(columns={'Lscore': 'team1Lmin'},
                                 inplace=True)
    grplosingteam1resultsmax = \
        losingteam1compactresults.groupby(['year', 'team1']).max()
    losingteam1resultsmax = grplosingteam1resultsmax.reset_index()
    losingteam1resultsmax.rename(columns={'Lscore': 'team1Lmax'},
                                 inplace=True)
    grplosingteam1resultsmedian = \
        losingteam1compactresults.groupby(['year', 'team1']).median()
    losingteam1resultsmedian = grplosingteam1resultsmedian.reset_index()
    losingteam1resultsmedian.rename(columns={'Lscore': 'team1Lmedian'},
                                    inplace=True)
    grplosingteam1resultsstd = \
        losingteam1compactresults.groupby(['year', 'team1']).std()
    losingteam1resultsstd = grplosingteam1resultsstd.reset_index()
    losingteam1resultsstd.rename(columns={'Lscore': 'team1Lstd'},
                                 inplace=True)
    winningteam2compactresults = pd.merge(how='left',
                                          left=teamcompactresults2,
                                          right=regularseasoncompactresults,
                                          left_on=['year', 'team2'],
                                          right_on=['Season', 'Wteam'])
    winningteam2compactresults.drop(['Season',
                                     'Daynum',
                                     'Wteam',
                                     'Lteam',
                                     'Lscore',
                                     'Wloc',
                                     'Numot'],
                                    inplace=True,
                                    axis=1)
    grpwinningteam2resultsaverage = \
        winningteam2compactresults.groupby(['year', 'team2']).mean()
    winningteam2resultsaverage = grpwinningteam2resultsaverage.reset_index()
    winningteam2resultsaverage.rename(columns={'Wscore': 'team2WAverage'},
                                      inplace=True)
    grpwinningteam2resultsmin = \
        winningteam2compactresults.groupby(['year', 'team2']).min()
    winningteam2resultsmin = grpwinningteam2resultsmin.reset_index()
    winningteam2resultsmin.rename(columns={'Wscore': 'team2Wmin'},
                                  inplace=True)
    grpwinningteam2resultsmax = \
        winningteam2compactresults.groupby(['year', 'team2']).max()
    winningteam2resultsmax = grpwinningteam2resultsmax.reset_index()
    winningteam2resultsmax.rename(columns={'Wscore': 'team2Wmax'},
                                  inplace=True)
    grpwinningteam2resultsmedian = \
        winningteam2compactresults.groupby(['year', 'team2']).median()
    winningteam2resultsmedian = grpwinningteam2resultsmedian.reset_index()
    winningteam2resultsmedian.rename(columns={'Wscore': 'team2Wmedian'},
                                     inplace=True)
    grpwinningteam2resultsstd = \
        winningteam2compactresults.groupby(['year', 'team2']).std()
    winningteam2resultsstd = grpwinningteam2resultsstd.reset_index()
    winningteam2resultsstd.rename(columns={'Wscore': 'team2Wstd'},
                                  inplace=True)
    losingteam2compactresults = pd.merge(how='left',
                                         left=teamcompactresults2,
                                         right=regularseasoncompactresults,
                                         left_on=['year', 'team2'],
                                         right_on=['Season', 'Lteam'])
    losingteam2compactresults.drop(['Season',
                                    'Daynum',
                                    'Wteam',
                                    'Lteam',
                                    'Wscore',
                                    'Wloc',
                                    'Numot'],
                                   inplace=True,
                                   axis=1)
    grplosingteam2resultsaverage = \
        losingteam2compactresults.groupby(['year', 'team2']).mean()
    losingteam2resultsaverage = grplosingteam2resultsaverage.reset_index()
    losingteam2resultsaverage.rename(columns={'Lscore': 'team2LAverage'},
                                     inplace=True)
    grplosingteam2resultsmin = \
        losingteam2compactresults.groupby(['year', 'team2']).min()
    losingteam2resultsmin = grplosingteam2resultsmin.reset_index()
    losingteam2resultsmin.rename(columns={'Lscore': 'team2Lmin'},
                                 inplace=True)
    grplosingteam2resultsmax = \
        losingteam2compactresults.groupby(['year', 'team2']).max()
    losingteam2resultsmax = grplosingteam2resultsmax.reset_index()
    losingteam2resultsmax.rename(columns={'Lscore': 'team2Lmax'},
                                 inplace=True)
    grplosingteam2resultsmedian = \
        losingteam2compactresults.groupby(['year', 'team2']).median()
    losingteam2resultsmedian = grplosingteam2resultsmedian.reset_index()
    losingteam2resultsmedian.rename(columns={'Lscore': 'team2Lmedian'},
                                    inplace=True)
    grplosingteam2resultsstd = \
        losingteam2compactresults.groupby(['year', 'team2']).std()
    losingteam2resultsstd = grplosingteam2resultsstd.reset_index()
    losingteam2resultsstd.rename(columns={'Lscore': 'team2Lstd'},
                                 inplace=True)
    agg_results = pd.merge(how='left',
                           left=merged_results,
                           right=winningteam1resultsaverage,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam1resultsaverage,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam1resultsmin,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam1resultsmin,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam1resultsmax,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam1resultsmax,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam1resultsmedian,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam1resultsmedian,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam1resultsstd,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam1resultsstd,
                           left_on=['year', 'team1'],
                           right_on=['year', 'team1'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam2resultsaverage,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam2resultsaverage,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam2resultsmin,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam2resultsmin,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam2resultsmax,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam2resultsmax,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam2resultsmedian,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam2resultsmedian,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=winningteam2resultsstd,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    agg_results = pd.merge(how='left',
                           left=agg_results,
                           right=losingteam2resultsstd,
                           left_on=['year', 'team2'],
                           right_on=['year', 'team2'])
    return agg_results


def GrabData():
    tourneyresults = pd.read_csv('files/TourneyCompactResults.csv')
    tourneyseeds = pd.read_csv('files/TourneySeeds.csv')
    regularseasoncompactresults = \
        pd.read_csv('files/RegularSeasonCompactResults.csv')
    sample = pd.read_csv('files/SampleSubmission.csv')
    results = pd.DataFrame()
    results['year'] = tourneyresults.Season
    results['team1'] = np.minimum(tourneyresults.Wteam, tourneyresults.Lteam)
    results['team2'] = np.maximum(tourneyresults.Wteam, tourneyresults.Lteam)
    results['result'] = (tourneyresults.Wteam <
                         tourneyresults.Lteam).astype(int)
    merged_results = pd.merge(left=results,
                              right=tourneyseeds,
                              left_on=['year', 'team1'],
                              right_on=['Season', 'Team'])
    merged_results.drop(['Season', 'Team'], inplace=True, axis=1)
    merged_results.rename(columns={'Seed': 'team1Seed'}, inplace=True)
    merged_results = pd.merge(left=merged_results,
                              right=tourneyseeds,
                              left_on=['year', 'team2'],
                              right_on=['Season', 'Team'])
    merged_results.drop(['Season', 'Team'], inplace=True, axis=1)
    merged_results.rename(columns={'Seed': 'team2Seed'}, inplace=True)
    merged_results['team1Seed'] = \
        merged_results['team1Seed'].apply(lambda x: str(x)[1:3])
    merged_results['team2Seed'] = \
        merged_results['team2Seed'].apply(lambda x: str(x)[1:3])
    merged_results = merged_results.astype(int)
    winsbyyear = regularseasoncompactresults[['Season', 'Wteam']].copy()
    winsbyyear['wins'] = 1
    wins = winsbyyear.groupby(['Season', 'Wteam']).sum()
    wins = wins.reset_index()
    lossesbyyear = regularseasoncompactresults[['Season', 'Lteam']].copy()
    lossesbyyear['losses'] = 1
    losses = lossesbyyear.groupby(['Season', 'Lteam']).sum()
    losses = losses.reset_index()
    winsteam1 = wins.copy()
    winsteam1.rename(columns={'Season': 'year',
                              'Wteam': 'team1',
                              'wins': 'team1wins'}, inplace=True)
    winsteam2 = wins.copy()
    winsteam2.rename(columns={'Season': 'year',
                              'Wteam': 'team2',
                              'wins': 'team2wins'}, inplace=True)
    lossesteam1 = losses.copy()
    lossesteam1.rename(columns={'Season': 'year',
                                'Lteam': 'team1',
                                'losses': 'team1losses'}, inplace=True)
    lossesteam2 = losses.copy()
    lossesteam2.rename(columns={'Season': 'year',
                                'Lteam': 'team2',
                                'losses': 'team2losses'}, inplace=True)
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=winsteam1,
                              left_on=['year', 'team1'],
                              right_on=['year', 'team1'])
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=lossesteam1,
                              left_on=['year', 'team1'],
                              right_on=['year', 'team1'])
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=winsteam2,
                              left_on=['year', 'team2'],
                              right_on=['year', 'team2'])
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=lossesteam2,
                              left_on=['year', 'team2'],
                              right_on=['year', 'team2'])
    teamcompactresults1 = merged_results[['year', 'team1']].copy()
    teamcompactresults2 = merged_results[['year', 'team2']].copy()

    train = Aggregate(teamcompactresults1,
                      teamcompactresults2,
                      merged_results,
                      regularseasoncompactresults)

    sample['year'] = sample.Id.apply(lambda x: str(x)[:4]).astype(int)
    sample['team1'] = sample.Id.apply(lambda x: str(x)[5:9]).astype(int)
    sample['team2'] = sample.Id.apply(lambda x: str(x)[10:14]).astype(int)

    merged_results = pd.merge(how='left',
                              left=sample,
                              right=tourneyseeds,
                              left_on=['year', 'team1'],
                              right_on=['Season', 'Team'])
    merged_results.drop(['Season', 'Team'], inplace=True, axis=1)
    merged_results.rename(columns={'Seed': 'team1Seed'}, inplace=True)
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=tourneyseeds,
                              left_on=['year', 'team2'],
                              right_on=['Season', 'Team'])
    merged_results.drop(['Season', 'Team'], inplace=True, axis=1)
    merged_results.rename(columns={'Seed': 'team2Seed'}, inplace=True)
    merged_results['team1Seed'] = \
        merged_results['team1Seed'].apply(lambda x: str(x)[1:3]).astype(int)
    merged_results['team2Seed'] = \
        merged_results['team2Seed'].apply(lambda x: str(x)[1:3]).astype(int)
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=winsteam1,
                              left_on=['year', 'team1'],
                              right_on=['year', 'team1'])
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=lossesteam1,
                              left_on=['year', 'team1'],
                              right_on=['year', 'team1'])
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=winsteam2,
                              left_on=['year', 'team2'],
                              right_on=['year', 'team2'])
    merged_results = pd.merge(how='left',
                              left=merged_results,
                              right=lossesteam2,
                              left_on=['year', 'team2'],
                              right_on=['year', 'team2'])

    teamcompactresults1 = merged_results[['year', 'team1']].copy()
    teamcompactresults2 = merged_results[['year', 'team2']].copy()

    test = Aggregate(teamcompactresults1,
                     teamcompactresults2,
                     merged_results,
                     regularseasoncompactresults)

    return train, test



train, test = GrabData()

target = train.result.values
pd.DataFrame(train).to_csv("train_przed_usunieciem_kolumny.csv", index=True)
# usuniecie kolumny z wynikiem 
train.drop('result', inplace=True, axis=1)
# puste miejsca na -1
train.fillna(-1, inplace=True)

testids = test.Id.values
test.drop(['Id', 'Pred'], inplace=True, axis=1)
test.fillna(1, inplace=True)
standardScaler = StandardScaler()

# Zaokrąglenie danych do 4 miejsc po przecinku z dopasowanej tablicy
# fit_transform - obliczanie wartości oczekiwanej oraz obliczanie standaryzacji i dopasowanie do zbioru
pd.DataFrame(train).to_csv("train_przed_dopasowaniem.csv", index=True)
train[train.columns] = np.round(standardScaler.fit_transform(train), 4)
pd.DataFrame(train).to_csv("train_po_dopasowaniu.csv", index=True)
# transform - dokonuje standaryzacji 
pd.DataFrame(test).to_csv("test_przed_dopasowaniem.csv", index=True)
test[test.columns] = np.round(standardScaler.transform(test), 4)
pd.DataFrame(test).to_csv("test_po_dopasowaniu.csv", index=True)
pd.DataFrame(target).to_csv("target_przed_porowynaiem.csv", index=True)
print('Training...')
# tworzymy drzewo decyzyjne
# n_estimators - liczba drzew w lesie
# max_features - liczba kolumn 
# min_samples_split - minimalna liczba wymagana do rozdzielenia wezla (domyslnie 2)
# min_samples_leaf - minimalna liczba wymagana do liscia (domyslnie 1)
# max_depth - maksymalna glebokosc drzewa (domyślnie None)
# n_jobs - ile watkow - jezeli -1 to liczba rdzeni
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
extraTreesClasiifier = ExtraTreesClassifier(n_estimators=50,max_features = 27,criterion= 'entropy',min_samples_split= 1,
                            max_depth= 10, min_samples_leaf= 1, n_jobs = -1)      

# dopasowanie drzewa.
extraTreesClasiifier.fit(train,target) 
# przwidywanie dla zbioru treningowego
x_pred = extraTreesClasiifier.predict_proba(train)
pd.DataFrame(x_pred).to_csv("x_pred_przed_zaraz_przed_trenigu.csv", index=True)
new_x_pred = x_pred[:,1]*1.721
pd.DataFrame(new_x_pred).to_csv("x_pred_przed_zaraz_po_trenigu.csv", index=True)
new_x_pred = np.clip(new_x_pred, 1e-6, 1-1e-6)
pd.DataFrame(new_x_pred).to_csv("magia_z_pred.csv", index=True)
# pd.DataFrame(x_pred).to_csv("x_pred.csv", index=True)
# pd.DataFrame(X_train).to_csv("x_train.csv", index=True)
# np.clip - ogranicza tablice do a_min i a_max
# np.array[:,1] - wycięcie tylko drugiej kolumny (index = 1) kolumny
# TODO wydrukować odpowiednie wyniki z x_pred z wyciągniętą kolumną
# log_loss tablica z wartościami oczekiwanymi, tablica przewidzana przez precict_proba
# *1.721, 1e-6, 1-1e-6
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
print(log_loss(target, np.clip(x_pred[:,1]*1.721, 1e-6, 1-1e-6)))
print('Predict...')
# Przewidywanie wyników dla zestawu testowego. 
y_pred = extraTreesClasiifier.predict_proba(test)
#print(y_pred)
namesCol = ["Team1", "Team2"]
submission2 = pd.DataFrame(y_pred)
submission2.columns = namesCol
# submission.to_csv('test.csv', index=True)
submission = pd.DataFrame({'Id': testids,
                           'Pred': np.clip(y_pred[:,1], 1e-7, 1-1e-7),
                           'Team1Pred': submission2["Team1"],
                           'Team2Pred': submission2["Team2"]})
submission.to_csv('submission.csv', index=True)
# print(submission2)
print('Finished')
# *1.0691   