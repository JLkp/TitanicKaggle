import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

SEED = 42
np.random.seed(SEED)


pd.set_option('display.max_columns', 20)

plt.figure(dpi=1200)
sns.set_theme("notebook", rc={"figure.dpi": 1200, 'figure.figsize': (10, 8)})

# Raw Daten einlesen
train_raw = pd.read_csv('train.csv', sep=',')
test_raw = pd.read_csv('test.csv', sep=',')



# Survived
survived = train_raw.Survived


# =============================================================================
# Pclass untersuchen
# =============================================================================
def explore_pclass():
    """Erkunden Pclass"""
    pclass = train_raw.Pclass

    pclass1 = pclass[pclass == 1]
    survived_pclass1 = survived.iloc[pclass1.index]
    survived_pclass1.plot.density(label='pclass1')

    pclass2 = pclass[pclass == 2]
    survived_pclass2 = survived.iloc[pclass2.index]
    survived_pclass2.plot.density(label='pclass2')

    pclass3 = pclass[pclass == 3]
    survived_pclass3 = survived.iloc[pclass3.index]
    survived_pclass3.plot.density(label='pclass3')
    plt.legend()
    plt.show()

# =============================================================================
# Name untersuchen
# =============================================================================
name = train_raw.Name
name_split = name.str.split(' ')
name_only_title = []
for full_name in name_split:
    name_sex = full_name[1]
    name_only_title.append(name_sex)

name_only_title = np.array(name_only_title)
name_only_title = pd.Series(name_only_title)
name_only_title_count = np.array(name_only_title.value_counts())

def plot_count_unique_names():
    """Plot Anzahl unique Namen"""
    fig_name, ax_1 = plt.subplots(1,1)

    ax_1.barh(name_only_title.unique(), name_only_title_count)
    ax_1.set_xlabel('Anzahl')
    ax_1.set_ylabel('Unique Namen')

    plt.show()


# In Feature Name alle uniquen Title (Mr, Mrs, Miss) in Zahlen renamen
# Title in Gruppen aufteilen
# männlich --> 0 (Mr., Master., Don., Rev., Dr., Col., Capt.)
male = ['Mr.', 'Master.', 'Don.', 'Rev.', 'Dr.', 'Col.', 'Capt.', 'Major.']
for title in male:
    name_only_title = name_only_title.replace(title, 0)

# weiblich --> 1 (Mrs., Miss., Mme.)
female = ['Mrs.', 'Miss.', 'Mme.', 'Ms.']
for title in female:
    name_only_title = name_only_title.replace(title, 1)

# sonstige --> 2 (Restlichen Namen)
name_rest = ['Planke,', 'Billiard,', 'der', 'Walle,', 'Pelsmaeker,', 'Mulder,',
             'y', 'Steen,', 'Carlo,', 'Impe,', 'Ms.' 'Major.', 'Gordon,', 
             'Messemaeker,', 'Mlle.', 'Velde,', 'the', 'Shawah,', 'Jonkheer.', 
             'Melkebeke,', 'Cruyssen,']
for title in name_rest:
    name_only_title = name_only_title.replace(title, 2)


# name_title als Feature in train_raw hinzufügen und Name droppen
train_raw['NameTitle'] = name_only_title
train_raw = train_raw.drop('Name', axis=1)

# Name mit survived kombinieren und histogramm erstellen
def hist_gender_survived():
    """Ploten Histogramm für Sex und Survived"""
    male = name_only_title[name_only_title == 0]
    male_survived_combined = survived.iloc[male.index]

    female = name_only_title[name_only_title == 1]
    female_survived_combined = survived.iloc[female.index]

    rest = name_only_title[name_only_title == 2]
    rest_survived_combined = survived[rest.index]

    fig_name_survived, (ax0, ax1, ax2) = plt.subplots(3,1)

    ax0.hist(male_survived_combined)
    ax0.set_title('Male Survived Combined')

    ax1.hist(female_survived_combined)
    ax1.set_title('Female Survived Combined')

    ax2.hist(rest_survived_combined)
    ax2.set_title('Rest Survived Combined')

    plt.show()


# =============================================================================
# Sex untersuchen
# =============================================================================

# Male/Female mit 1/0 replacen
train_raw.Sex = train_raw.Sex.replace('male', 1)
train_raw.Sex = train_raw.Sex.replace('female', 0)

sex = pd.Series(train_raw.Sex)
def pie_chart_sex_survived():
    """Pie Chart für Se und Survived"""
    fig_sex, (ax1_sex, ax2_sex) = plt.subplots(1,2)
    # Pie Chart Male Survivev/Died
    male = sex[sex == 1]
    male_and_survived = survived.iloc[male.index]
    male_survived = male_and_survived[male_and_survived == 1]
    male_survived_percent = (male_survived.shape[0] / male.shape[0]) * 100
    male_died_percent = 100 - male_survived_percent
    ax1_sex.pie(x=[male_survived_percent, male_died_percent],
                labels=['Survived', 'Died'],
                autopct='%1.1f%%')
    ax1_sex.set_title('Male')

    # Pie Chart Female Survived/Died
    female = sex[sex == 0]
    female_and_survived = survived.iloc[female.index]
    female_survived = female_and_survived[female_and_survived == 1]
    female_survived_percent = (female_survived.shape[0] / female.shape[0]) * 100
    female_died_percent = 100 - female_survived_percent
    ax2_sex.pie(x=[female_survived_percent, female_died_percent],
                labels=['Survived', 'Died'],
                autopct='%1.1f%%')
    ax2_sex.set_title('Female')
    plt.show()


# =============================================================================
# Age untersuchen
# =============================================================================

# fehlende Werte hinzufügen
train_raw['Age'] = train_raw.Age.fillna(train_raw.Age.mean())


age = pd.Series(train_raw.Age)

mean_age = age.mean()
median_age = age.median()
mode_age = age.mode()

# Means, Labels, clrs festlegen um in ax1_age Means hinzuzufügen
means = [age.mean(), age.median(), age.mode()[0]]
labels = ['Mean', 'Median', 'Mode']
clrs = ['red', 'blue', 'green']

fig_age, (ax1_age, ax2_age) = plt.subplots(2,1)
ax1_age.hist(age)
for idx, (mean, label_s, clr) in enumerate(zip(means, labels, clrs)):
    print(idx, mean, label_s, clr)
    ax1_age.axvline(x=mean,
                    label=label_s,
                    color=clr,
                    linestyle='--')



ax1_age.legend()
ax2_age.boxplot(age, vert=False)
plt.show()

# =============================================================================
# Age und Sex untersuchen/plotten
# =============================================================================
def male_female_age_plot():
    """Plotten von Sex und Age"""
    male_only = sex[sex == 1]
    male_and_age = age[male_only.index]

    fig_male_age, (ax0_male_age, ax1_male_age) = plt.subplots(2,1)
    ax0_male_age.hist(male_and_age)
    ax0_male_age.set_title('Hist Male and Age')

    ax1_male_age.boxplot(male_and_age, vert=False)
    ax1_male_age.set_title('Boxplot Male and Age')
    plt.show()

    iqr_male_and_age = 35 - 23 # 3.Quartile - 1.Quartile
    lower_whisker_male_age = 23 - (1.5* iqr_male_and_age)
    upper_whisker_male_age = 35 + (1.5 * iqr_male_and_age)
    print(f"Unterer Schwellenwert male_and_age: {lower_whisker_male_age}")
    print(f"Oberer Schwellenwert male_and_age: {upper_whisker_male_age} \n")

    female_only = sex[sex == 0]
    female_and_age = age[female_only.index]

    fig_female_age, (ax0_female_age, ax1_female_age) = plt.subplots(2,1)
    ax0_female_age.hist(female_and_age)
    ax0_female_age.set_title('Hist Female and Age')

    ax1_female_age.boxplot(female_and_age, vert=False)
    ax1_female_age.set_title('Boxplot Female and Age')
    plt.show()

    iqr_female_and_age = 35 - 21 # 3.Quartile - 1.Quartile
    lower_whisker_female_age = 21 - (1.5 * iqr_female_and_age)
    upper_whisker_female_age = 35 + (1.5 * iqr_female_and_age)
    print(f"Unterer Schwellenwert female_and_age: {lower_whisker_female_age}")
    print(f"Oberer Schwellenwert female_and_age: {upper_whisker_female_age}")

male_female_age_plot()

# Schwellenwerte von Age
IQR_AGE = 35 - 22
LOWER_WHISKER_AGE = 22 - (1.5 * IQR_AGE)  # 2.5
UPPER_WHISKER_AGE = 35 + (1.5 * IQR_AGE)  # 54.5

# in male_age/female_age Outlier bestimmen
sex = train_raw.Sex

male_only = sex[sex == 1]
male_and_age = age[male_only.index]
male_age_outlier_lower = male_and_age[age < 3] #  kleinerer Schwellenwert als Whisker
male_age_outlier_upper = male_and_age[age > 65] # höherer Schwellenwert als Whisker
male_age_outlier = pd.concat([male_age_outlier_lower,
                             male_age_outlier_upper]
                             ) # Insgesamt 22 Outlier


female_only = sex[sex == 0]
female_and_age = age[female_only.index]
female_age_outlier = female_and_age[age > 56] # Insgesamt 9 Outlier

male_female_age_outlier = pd.concat([male_age_outlier,
                                     female_age_outlier])
# sex_age Outlier dropen
train_raw = train_raw.drop(male_female_age_outlier.index, axis=0)

# =============================================================================
# SibSp untersuchen
# =============================================================================
sibsp = train_raw.SibSp
fig_sibsp, (ax1_sibsp, ax2_sibsp) = plt.subplots(2,1)
ax1_sibsp.hist(sibsp)
ax1_sibsp.set_title('Histogramm SibSp')

ax2_sibsp.boxplot(sibsp, vert=False)
ax1_sibsp.set_title('Boxplot SibSp')
plt.show()

IQR_SIBSP = 1 - 0
LOWER_WHISKER_SIBSP = 0 - (1.5 * IQR_SIBSP) # -1
UPPER_WHISKER_SIBSP = 1 + (1.5 * IQR_SIBSP) # 2.5
sibsp_outliers_upper = sibsp[sibsp > 4] # höherer Schwellenwert als Whisker

# sibsp outlier dropen
train_raw = train_raw.drop(sibsp_outliers_upper.index, axis=0)

# =============================================================================
# Parch untersuchen
# =============================================================================
parch = train_raw.Parch
fig_parch, (ax1_parch, ax2_parch) = plt.subplots(2,1)
ax1_parch.hist(parch)
ax1_parch.set_title('Histogramm Parch')

ax2_parch.boxplot(parch, vert=False)
ax2_parch.set_title('Boxplot Parch')
plt.show()

parch_bigger_4 = parch[parch > 4] # 4 sinnvoller Schwellenwert

# parch outlier dropen
train_raw = train_raw.drop(parch_bigger_4.index, axis=0)


# =============================================================================
# Ticket untersuchen
# =============================================================================

ticket = train_raw.Ticket

le = LabelEncoder()
ticket_transformed = le.fit_transform(ticket)

fig_ticket, (ax1_ticket, ax2_ticket) = plt.subplots(2,1)
ax1_ticket.hist(ticket_transformed)
ax1_ticket.set_title('Histogramm Ticket Transformed')

ax2_ticket.boxplot(ticket_transformed, vert=False)
ax2_ticket.set_title('Boxplot Ticket Transformed')
plt.show()

# Ticket dropen und TicketTransformed als Feature hinzufügen
train_raw = train_raw.drop('Ticket', axis=1)
train_raw['TicketTransformed'] = ticket_transformed

# survived neu inizialisieren, da Rows gelöscht wurden
survived = train_raw.Survived
# =============================================================================
# Fare untersuchen
# =============================================================================
fare = train_raw.Fare

# Korrelation zwischen Fare und Survived bestimmen --> nicht auffällig
corr_fare_survived, p_fare_survived = stats.pearsonr(x=fare, y=survived)
print(f"Korrelation Fare Survived: {corr_fare_survived}, Statistische Relevanz: {p_fare_survived}")


fig_fare, (ax1_fare, ax2_fare) = plt.subplots(2,1)
ax1_fare.hist(fare)
ax1_fare.set_title("Histogramm Fare")

ax2_fare.boxplot(fare, vert=False)
ax2_fare.set_title("Boxplot Fare")
plt.show()

fare_outliers = fare[fare > 400]

# Fare Outliers droppen
train_raw = train_raw.drop(fare_outliers.index, axis=0)

# =============================================================================
# Embarked untersuchen
# =============================================================================
train_raw.Embarked = train_raw.Embarked.fillna('S')

le = LabelEncoder()
train_raw['Embarked'] = le.fit_transform(train_raw.Embarked)


# =============================================================================
# Cabin untersuchen --> Clustern, da viele fehlende Werte
# =============================================================================

train_raw_only_numeric = train_raw.select_dtypes('number')


pca = PCA(n_components=3)
train_2_features = pca.fit_transform(train_raw_only_numeric)

fig_2_features = plt.figure(figsize=(10,15))
ax1_2_features = fig_2_features.add_subplot(projection='3d')
ax1_2_features.scatter(train_2_features[:,0],
                       train_2_features[:,1],
                       train_2_features[:,2],
                       marker='o')
ax1_2_features.set_xlabel('Feature 0')
ax1_2_features.set_ylabel('Feature 1')
ax1_2_features.set_zlabel('Feature 2')
ax1_2_features.set_title('Train Data PCA 2 Features')
plt.show()

df_fit_cabin = train_raw.drop('Cabin', axis=1)
df_fit_cabin = ((df_fit_cabin - df_fit_cabin.min()) /
                (df_fit_cabin.max() - df_fit_cabin.min()))

print(df_fit_cabin.NameTitle)

def test_kmeans_wcss(): # beste Clusteranzahl ist 4
    """Bestimmen von bester Anzahl der Cluster für Cabin"""
    wcss = []
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i, max_iter=1000, n_init=200).fit(df_fit_cabin)
        wcss.append(kmeans.inertia_)

    wcss = np.array(wcss)
    plt.plot(wcss)
    plt.show()

test_kmeans_wcss()


# cabin mit kmeans clustern
kmeans = KMeans(n_clusters=4, max_iter=1000, n_init=200).fit(df_fit_cabin)
cabin_clustered = kmeans.labels_

# cabin_clustered in cabin in train_raw einsetzten
train_raw['Cabin'] = cabin_clustered

# train_raw normalisieren

scaler = MinMaxScaler(feature_range=(0,1))
data_normalized = scaler.fit_transform(train_raw)
train_normalized = pd.DataFrame(data=data_normalized, columns=train_raw.columns)
train_normalized['PassengerId'] = train_raw['PassengerId']

train_normalized.to_csv('train_normalized.csv', sep=',', index=False)
train_raw.to_csv('train_standard.csv', sep=',', index=False)

# gleichen Prozess für test_raw, nur kombinieren, name in title minimieren,
# age fehlende mit mean auffüllen, Embarked encoden, cabin clustern,
