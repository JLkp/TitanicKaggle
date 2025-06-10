import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

SEED = 42
np.random.seed(SEED)


pd.set_option('display.max_columns', 20)

plt.figure(dpi=1200)
sns.set_theme("notebook", rc={"figure.dpi": 1200, 'figure.figsize': (10, 8)})


# Test Data importieren
df = pd.read_csv('test.csv', sep=',')
print(df.columns)

print(df.isna().sum()) # Age(86), Fare(1) und Cabin(327) haben fehlende Werte


# Pclass untersuchen
pclass_1 = df.Pclass[df.Pclass == 1].count()
pclass_2 = df.Pclass[df.Pclass == 2].count()
pclass_3 = df.Pclass[df.Pclass == 3].count()
pclass_counts = np.array([pclass_1, pclass_2, pclass_3])
y_pos = np.arange(pclass_counts.shape[0])


fig_pclass, ax1_pclass = plt.subplots(1,1)
ax1_pclass.barh(y_pos, pclass_counts)
ax1_pclass.set_title("Counts der Pclass 1/2/3")
plt.show()


# Name untersuchen
name = df.Name

# Counts der uniquen Namen erzeugen und ploten
name_split = name.str.split(' ')
name_title = []
for name in name_split:
    name_title.append(name[1])
name_title = pd.Series(np.array(name_title))
name_title_count = name_title.value_counts()


fig_name, ax1_name = plt.subplots(1,1) 
ax1_name.barh(name_title.unique(), name_title_count)
ax1_name.set_title("Counts der uniquen Namen")
plt.show()

male = ['Mr.', 'Master.', 'Col.', 'Rev.', 'Billiard,', 'Carlo,', 'Dr.']
female = ['Miss.', 'Ms.', 'Mrs.']
other = ['y', 'Khalil,', 'Palmquist,', 'Planke,', 'Messemaeker,', 'Brito,']


# Alle Title aus male, female, other mit 1,2,3 replacen
for title in male:
    name_title = name_title.replace(title, 0)

for title in female:
    name_title = name_title.replace(title, 1)

for title in other:
    name_title = name_title.replace(title, 2)

df = df.drop("Name", axis=1)
df["NameTitle"] = name_title

# Sex untersuchen
df.Sex = df.Sex.replace("male", 0)
df.Sex = df.Sex.replace("female", 1)

# Age untersuchen
df.Age = df.Age.fillna(df.Age.mean())

# SibSp untersuchen
sibsp = df.SibSp
fig_sibsp, (ax1_sibsp, ax2_sibsp) = plt.subplots(2,1)
ax1_sibsp.hist(sibsp)
ax1_sibsp.set_title("Histogramm SibSp")

ax2_sibsp.boxplot(sibsp, vert=False)
ax2_sibsp.set_title("Boxplot SibSp")
plt.show()

# Parch untersuchen

fig_parch, (ax1_parch, ax2_parch) = plt.subplots(2,1)
ax1_parch.hist(df.Parch)
ax1_parch.set_title("Histogramm Parch")

ax2_parch.boxplot(df.Parch, vert=False)
ax2_parch.set_title("Boxplot Parch")
plt.show()

# Ticket untersuchen
le = LabelEncoder()
ticket_transformed = le.fit_transform(df.Ticket)

df['TicketTransformed'] = ticket_transformed
df = df.drop("Ticket", axis=1)

# Fare untersuchen
df.Fare = df.Fare.fillna(df.Fare.mean())

fig_fare, (ax1_fare, ax2_fare) = plt.subplots(2,1)
ax1_fare.hist(df.Fare)
ax1_fare.set_title("Histogramm Fare")

ax2_fare.boxplot(df.Fare, vert=False)
ax2_fare.set_title("Boxplot Fare")
plt.show()

df.Fare.plot.density()

# Embarked untersuchen
df.Embarked = le.fit_transform(df.Embarked)

# Cabin untersuchen und clustern
df_fit_cabin = df.drop(["Cabin", "PassengerId"], axis=1)

def find_n_clusters(): # n_clusters=2 am besten 
    wcss = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters=i, max_iter=1000, n_init=200).fit(df_fit_cabin)
        wcss.append(kmeans.inertia_)
    
    plt.plot(wcss)


used_kmeans = KMeans(n_clusters=2, max_iter=1000, n_init=200).fit(df_fit_cabin)
cabin_clustered = used_kmeans.labels_

df['Cabin'] = cabin_clustered

# DataFrame auf 0 bis 1 normalisieren
scaler = MinMaxScaler(feature_range=(0,1))
features_normalized = scaler.fit_transform(df)
df_normalized = pd.DataFrame(data=features_normalized, columns=df.columns)
df_normalized["PassengerId"] = df.PassengerId

df_normalized.to_csv("test_normalized.csv", sep=',', index=False)
df.to_csv('test_standard.csv', sep=',', index=False)


