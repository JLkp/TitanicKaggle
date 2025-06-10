import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from scipy import stats

SEED = 42
np.random.seed(SEED)


pd.set_option('display.max_columns', 20)

plt.figure(dpi=1200)
plt.style.use('seaborn-v0_8-notebook')
sns.set_theme("notebook", rc={"figure.dpi": 1200, 'figure.figsize': (10, 8)})



# Daten einlesen
df_train = pd.read_csv('train.csv', sep=',')
df_test = pd.read_csv('test.csv', sep=',')

# Fehlende Werte in Train und Test
print(df_train.isna().sum())

# Pclass untersuchen: Verteilung der Klasssen, Survived/Pclass
pclass = df_train.Pclass
unique_pclass = np.sort(pclass.unique())


distribution_pclass = np.array([pclass.eq(unique_pclass[0]).sum(),
                                pclass.eq(unique_pclass[1]).sum(),
                                pclass.eq(unique_pclass[2]).sum()])

distribution_pclass_survived = np.array([df_train[df_train.Pclass == 1].query('Survived == 1').Pclass.count(),
                                         df_train[df_train.Pclass == 2].query('Survived == 1').Pclass.count(),
                                         df_train[df_train.Pclass == 3].query('Survived == 1').Pclass.count()])



fig_pclass, (ax1_pclass) = plt.subplots(1,1)
ax1_pclass.bar(unique_pclass, 
               distribution_pclass, 
               label='Sum of Class',)
ax1_pclass.bar(unique_pclass, 
               distribution_pclass_survived, 
               label='Survived', 
               alpha=0.8)
ax1_pclass.set_title('Anzahl der Personen pro Klasse')
plt.legend()
plt.show()


# Pclass mit OneHotEncoder encoden, Pclass dropen
pclass_onehot = pd.get_dummies(pclass, dtype=int)
df_train['Pclass_1'] = pclass_onehot.loc[:,1]
df_train['Pclass_2'] = pclass_onehot.loc[:,2]
df_train['Pclass_3'] = pclass_onehot.loc[:,3]

# df_train = df_train.drop('Pclass', axis=1)


# Name untersuchen, dropen --> schwer aus Name gute Info zu erhalten
df_train = df_train.drop('Name', axis=1)

# Sex untersuchen: Anteil Überlebende pro Geschlecht, Geschlechterverteilung
#                  in 0 (male), 1 (female) encoden  

df_train.Sex = df_train.Sex.replace(['male', 'female'], [0,1])
sex = df_train.Sex


# Geschlechterverteilung
distribution_sex = np.array([sex[sex == 0].count(), sex[sex == 1].count()])

# Anteil der Überlebenden pro Geschlecht
distribution_sex_survived = np.array([df_train[df_train.Sex == 0].query('Survived == 1').Sex.count(),
                                      df_train[df_train.Sex == 1].query('Survived == 1').Sex.count()])

colors = ['b', 'm']
fig_sex, (ax1_sex, ax2_sex) = plt.subplots(1,2)
ax1_sex.pie(distribution_sex, 
            labels=['Male', 'Female'], 
            autopct='%1.1f%%',
            colors=colors)
ax1_sex.set_title('Anteil Männlich, Weiblich')

ax2_sex.bar(['Male','Female'], distribution_sex, label='Anzahl Geschlecht')
ax2_sex.bar(['Male','Female'], 
            distribution_sex_survived, 
            label='Anzahl Überlebende Geschlecht', 
            alpha=0.8,
            color='m')
plt.legend()
plt.show()


def compare_numeric(feature_train, feature_test):
    """
    feature_train: Feature in Trainings Datensatz
    feature_test: Feature in Test Datensatz
    
    Funktion vergleicht Features mit Histogramm
    """
    
    fig_compare, ax1_compare = plt.subplots(1,1)
    fig_compare.suptitle(feature_train.name)
    ax1_compare.hist(feature_train)
    ax1_compare.hist(feature_test)

    plt.show()


# Age, SibSp, Parch aus train und test vergleichen

# Age
compare_numeric(df_train.Age.fillna(df_train.Age.mean()), 
                df_test.Age.fillna(df_test.mean()))

# SibSp
compare_numeric(df_train.SibSp, df_test.SibSp)

# Parch in getrenten Plots vergleichen, da Feature in einem Plot nichts bringt
fig_parch, (ax1_parch, ax2_parch) = plt.subplots(2,1)
ax1_parch.hist(df_train.Parch)
ax1_parch.set_title('Histogramm Train Daten Parch')
ax2_parch.hist(df_test.Parch)
ax2_parch.set_title('Histogramm Test Daten Parch')
plt.show()


# SibSp untersuchen: Verteilung plotten, Korrelation mit Survived, One Hot Encoding
fig_SibSp , (ax1_SibSp, ax2_SibSp) = plt.subplots(2,1)
ax1_SibSp.hist(df_test.SibSp)
ax1_SibSp.set_title('Histogramm SibSp')

ax2_SibSp.boxplot(df_test.SibSp, vert=False)
ax2_SibSp.set_title('Boxplot SibSp')
plt.show()

# Outlier ab 5 dropen
SibSp_outlier_index = df_train.SibSp[df_train.SibSp >= 5].index
df_train = df_train.drop(SibSp_outlier_index, axis=0)

# df_train reindexen
df_train = df_train.reset_index(drop=True)


# SibSp mit OneHotEncoder neu encoden und SibSp dropen
SibSp_onehot = pd.get_dummies(df_train.SibSp, dtype=int)
df_train['SibSp_0'] = SibSp_onehot.loc[:,0]
df_train['SibSp_1'] = SibSp_onehot.loc[:,1]
df_train['SibSp_2'] = SibSp_onehot.loc[:,2]
df_train['SibSp_3'] = SibSp_onehot.loc[:,3]
df_train['SibSp_4'] = SibSp_onehot.loc[:,4]

df_train = df_train.drop('SibSp', axis=1)

corr_SibSp = np.array([np.corrcoef(df_train.Survived, df_train.SibSp_0),
                       np.corrcoef(df_train.Survived, df_train.SibSp_1),
                       np.corrcoef(df_train.Survived, df_train.SibSp_2),
                       np.corrcoef(df_train.Survived, df_train.SibSp_3),
                       np.corrcoef(df_train.Survived, df_train.SibSp_4)])

# Age untersuchen: Verteilung vom Alter, Unterteilung in Altersgruppen, 
#                  Wahrscheinlichkeit des Überlebens jeder Altergruppe, 
#                  Outlier dropen --> Vergleich mit Testdaten, nicht 
#                  zu viele dropen, NaN mit Mean auffüllen

# NaN mit Mean auffüllen
mean_age = df_train.Age.mean()
df_train.Age = df_train.Age.fillna(mean_age)

# Verteilung des Alters in Traindata und Testdata
fig_age_dist, (ax1_age_dist, ax2_age_dist, ax3_age_dist) = plt.subplots(3,1, 
                                                                        figsize=(10,12))
ax1_age_dist.hist(df_train.Age)
ax1_age_dist.hist(df_test.Age)
ax1_age_dist.set_title('Histogramm Age')

ax2_age_dist.boxplot(df_train.Age, vert=False, whis=1.5)
ax2_age_dist.set_title('Boxplot Train Age')

ax3_age_dist.boxplot(df_test.Age.fillna(mean_age), vert=False, whis=1.5)
ax3_age_dist.set_title('Boxplot Test Age')

plt.show()

# Werte außerhalb der Whisker von Test Daten dropen
IQR_AGE = df_test.Age.quantile(0.75) - df_test.Age.quantile(0.25)
lower_whisker = np.abs(df_test.Age.quantile(0.25) - (1.5 * IQR_AGE))
upper_whisker = np.abs(df_test.Age.quantile(0.75) + (1.5 * IQR_AGE))

lower_age_train_index = df_train.Age[df_train.Age < lower_whisker].index.to_numpy()
upper_age_train_index = df_train.Age[df_train.Age > upper_whisker].index.to_numpy()

outlier_age_index = np.append(lower_age_train_index, 
                              upper_age_train_index)

# Outlier aus df_train dropen und index reseten
df_train = df_train.drop(outlier_age_index, axis=0).reset_index(drop=True)

# Unterteilen in Gruppen, Wkeit fürs Überleben plotten

# 0-20
age_1 = df_train.Age[df_train.Age <= 10]

# 21-40
age_2 = df_train.loc[(df_train['Age'] > 10) & 
                     (df_train['Age'] <= 40)].Age

# 41-66
age_3 = df_train.Age[df_train.Age >= 41]

# Anzahl Überlebte pro Altersgruppe
age_1_survived = df_train[df_train.Age <= 10].query("Survived == 1").Survived.count()
age_2_survived = df_train.loc[age_2.index].query("Survived == 1").Survived.count()
age_3_survived = df_train[df_train.Age >= 41].query("Survived == 1").Survived.count()

age_1_died = age_1.count() - age_1_survived
age_2_died = age_2.count() - age_2_survived
age_3_died = age_3.count() - age_3_survived


fig_age_surv, (ax1_age_surv, ax2_age_surv, ax3_age_surv) = plt.subplots(1,3)
ax1_age_surv.pie([age_1_survived, age_1_died],
                 labels=['Survived', 'Died'],
                 autopct='%1.1f%%')
ax1_age_surv.set_title('0-10 Jahre')

ax2_age_surv.pie([age_2_survived, age_2_died],
                 labels=['Survived', 'Died'],
                 autopct='%1.1f%%')
ax2_age_surv.set_title('11-40 Jahre')

ax3_age_surv.pie([age_3_survived, age_3_died],
                 labels=['Survived', 'Died'],
                 autopct='%1.1f%%')
ax3_age_surv.set_title('41-80 Jahre')
plt.show()


# Parch untersuchen: Unique Values, Outlier angepasst an Testdaten dropen,
#                    Wkeit Überlebende für unique Values, vllt OneHotEncoden

fig_parch, (ax1_parch, ax2_parch) = plt.subplots(2,1)
ax1_parch.hist(df_train.Parch)
ax1_parch.hist(df_test.Parch)
ax1_parch.set_title('Histogramm Parch Train und Testdaten')

ax2_parch.boxplot(df_test.Parch, vert=False)
ax2_parch.set_title('Boxplot Parch Testdaten')
plt.show()

# Alles über 3 dropen
parch_outlier_index = df_train[df_train.Parch > 3].Parch.index
df_train = df_train.drop(parch_outlier_index, axis=0).reset_index(drop=True)

# Wkeit Überleben für unique Values
survived_unique_parch = df_train[df_train.Survived == 1].Parch.value_counts()
died_unique_parch = df_train[df_train.Survived == 0].Parch.value_counts()


fig_parch, [[ax0_parch, ax1_parch], [ax2_parch, ax3_parch]] = plt.subplots(2,2)
ax0_parch.pie([survived_unique_parch.iloc[0], died_unique_parch.iloc[0]],
              labels=['Survived', 'Died'],
              autopct='%1.1f%%')
ax0_parch.set_title('Survived/Died Parch = 0')

ax1_parch.pie([survived_unique_parch.iloc[1], died_unique_parch.iloc[1]],
              labels=['Survived', 'Died'],
              autopct='%1.1f%%')
ax1_parch.set_title('Survived/Died Parch = 1')

ax2_parch.pie([survived_unique_parch.iloc[2], died_unique_parch.iloc[2]],
              labels=['Survived', 'Died'],
              autopct='%1.1f%%')
ax2_parch.set_title('Survived/Died Parch = 2')

ax3_parch.pie([survived_unique_parch.iloc[3], died_unique_parch.iloc[3]],
              labels=['Survived', 'Died'],
              autopct='%1.1f%%')
ax3_parch.set_title('Survived/Died Parch = 3')
plt.show()

df_train.Parch = df_train.Parch.replace(3,2)


# Ticket untersuchen
# Ticket dropen --> vermutlich wenig Information in Feature, Ticket bzw. Ort 
# des an Bord Gehen unwichtig für das Überleben/Sterben des Pasagiers

df_train = df_train.drop('Ticket', axis=1)

# Fare untersuchen: Versuchen zu Clustern, mit Pclass ploten --> Zusammenhang 
#                   zwischen Pclass und Fare?, vermutlich 3 Cluster, für 
#                   Cluster Boxplot um durschn. Fare zu plotten


# Fare Scatter plotten
counts_fare = np.arange(df_train.Fare.shape[0])

fig_fare, ax1_fare = plt.subplots(1,1)
ax1_fare.scatter(counts_fare, df_train.Fare)
ax1_fare.set_title('Scatter Plot für Fare')
plt.show()

# mit kmeans Fare cluster
df_cluster_fare = df_train.drop(['Fare', 'Embarked', 'Cabin'], axis=1)
kmeans = KMeans(n_clusters=3, 
                n_init=20, 
                max_iter=1000, 
                random_state=SEED)
fare_clustered = kmeans.fit_predict(df_cluster_fare, df_train.Fare)

# fare_clustered in df_train hinzufügen und Fare dropen
df_train['FareClustered'] = fare_clustered

# Cabin untersuchen
print(f"Anzahl NA-Werte Cabin: {df_train.Cabin.isna().sum()}")

# Cabin dropen, da zuviele NA Werte 
df_train = df_train.drop('Cabin', axis=1)


# Embarked untersuchen: Mapping von catogorical Features, NA in Kategorie
map_embarked = {'S': 1,
                'C': 2,
                'Q': 3}

df_train.Embarked = df_train.Embarked.map(map_embarked)

# NA-Werte mit gleichem Wert bei FareEmbarked imputen
mask = df_train.Embarked.isna()

index_na = df_train.Embarked.index[mask].values
fare_imputer = df_train.FareClustered[index_na]

mean_embarked_imputer = np.array([np.floor(df_train.query('FareClustered == 0').Embarked.mean()),
                                  np.floor(df_train.query('FareClustered == 2').Embarked.mean())]
                                 )
# Da mean_embarked_imputer für beide Stellen gleich ist (1.0,1.0) wird fillna benutzt
df_train.Embarked = df_train.Embarked.fillna(1)

df_train = df_train.drop(['PassengerId', 'Pclass', 'Fare'], axis=1)

df_train_min_max = (df_train - df_train.min())/ (df_train.max() - df_train.min())

df_train_min_max.Survived = df_train.Survived

df_train_min_max.to_csv('min_max_train.csv', sep=',', index=False)

# =============================================================================
# Testdaten analysieren
# =============================================================================
df_test = pd.read_csv('test.csv', sep=',')

# Schlechte Features dropen
df_test = df_test.drop(['Ticket', 'Cabin', 'Name'], axis=1)


corr_matrix = df_test.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2g", cmap="crest").set_title('Corrmatrix Testdaten')

# Pclass: One-Hot-Encoden
pclass_dummies_test = pd.get_dummies(df_test.Pclass, prefix='Pclass', prefix_sep='_')

df_test = pd.concat([df_test, pclass_dummies_test], axis=1)


# Sex: Male und Female zu 0/1 mappen
df_test.Sex = df_test.Sex.replace(['male', 'female'], [0,1])


# Age: NA-Werte mit Mean-Wert von gleicher Pclass auffüllen
mean_age_pclass1 = int(df_test[df_test.Pclass == 1].Age.mean())
mean_age_pclass2 = int(df_test[df_test.Pclass == 2].Age.mean())
mean_age_pclass3 = int(df_test[df_test.Pclass == 3].Age.mean())

age_na_pclass1_index = df_test[df_test.Age.isna()].query('Pclass == 1').index
age_na_pclass2_index = df_test[df_test.Age.isna()].query('Pclass == 2').index
age_na_pclass3_index = df_test[df_test.Age.isna()].query('Pclass == 3').index


df_test.Age.loc[age_na_pclass1_index] = mean_age_pclass1
df_test.Age.loc[age_na_pclass2_index] = mean_age_pclass2
df_test.Age.loc[age_na_pclass3_index] = mean_age_pclass3

# Pclass dropen, da Feature ab hier überflüssig ist 
df_test = df_test.drop('Pclass', axis=1)


# SibSp: alles > 3 (4,5,8) zu einer Gruppe encoden, One-Hot-Encoden
df_test.SibSp = df_test.SibSp.replace([8,5] , 4)

SibSp_dummies_test = pd.get_dummies(df_test.SibSp, prefix='SibSp', prefix_sep='_')

df_test = pd.concat([df_test, SibSp_dummies_test], axis=1)

df_test = df_test.drop('SibSp', axis=1)


# Parch: alles > 3 zu einer Gruppe, alles = 3 zu 2 mappen (gleiche Wkeit zu überleben)
df_test.Parch = df_test.Parch.replace([5,6,9], 4)
df_test.Parch = df_test.Parch.replace(3, 2)


# Embarked: categorical zu numeric mappen
map_embarked = {'Q': 0,
                'S': 1,
                'C': 2}

df_test.Embarked = df_test.Embarked.map(map_embarked)


# Fare: NA-Werte mit mode imputen, mit kmeans (n_clusters=3) clustern
fare_mode = stats.mode(df_test.Fare, axis=None, keepdims=False)

df_test.Fare = df_test.Fare.fillna(fare_mode.mode)

df_cluster_fare_test = df_test.drop(['PassengerId', 'Fare'], axis=1)

fare_clustered_test = kmeans.fit_predict(df_cluster_fare_test, df_test.Fare)

df_test['FareClustered'] = fare_clustered_test

df_test = df_test.drop('Fare', axis=1)

df_test_min_max = (df_test - df_test.min()) / (df_test.max() - df_test.min())

df_test_min_max.PassengerId = df_test.PassengerId

df_test_min_max.to_csv('min_max_test.csv', sep=',', index=False)
