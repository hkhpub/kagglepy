import pandas as pd
import numpy as np

df = pd.read_csv('/home/hkh/sources/kagglepy/titanic/data/train.csv', header=0)

# cleaning the data
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1}).astype(int)

median_ages = np.zeros((2,3))

for i in range(0, 2):
	for j in range(0, 3):
		median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):
	for j in range(0, 3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

# clean Embarked column
df['EmbarkedFill'] = df['Embarked']
df.loc[df.Embarked.isnull(), 'EmbarkedFill'] = 'S'
df['EmbarkedFill'] = df['EmbarkedFill'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

# remove non-numerical columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.drop(['Age'], axis=1)
df = df.dropna()

train_data = df.values


##
## test data cleaning
df = pd.read_csv('/home/hkh/sources/kagglepy/titanic/data/test.csv', header=0)

# cleaning the data
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1}).astype(int)

median_ages = np.zeros((2,3))

for i in range(0, 2):
	for j in range(0, 3):
		median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):
	for j in range(0, 3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

# clean Embarked column
df['EmbarkedFill'] = df['Embarked']
df.loc[df.Embarked.isnull(), 'EmbarkedFill'] = 'S'
df['EmbarkedFill'] = df['EmbarkedFill'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

# remove non-numerical columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.drop(['Age'], axis=1)
df = df.dropna()

test_data = df.values
