import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv(r'C:\Users\jidub\OneDrive\Documents\internship\skillcraft\titanic\train.csv')
test_data = pd.read_csv(r'C:\Users\jidub\OneDrive\Documents\internship\skillcraft\titanic\test.csv')

# Display the first few rows and basic information about the training dataset
print(train_data.head())
print(train_data.info())


print(train_data.isnull().sum())

# Fill missing 'Age' with median age
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to high number of missing values
train_data.drop('Cabin', axis=1, inplace=True)

# Verify missing values have been handled
print(train_data.isnull().sum())

# Convert 'Sex' to numeric
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numeric
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

survival_rate = train_data['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.2%}")

plt.figure(figsize=(8, 6))
train_data['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Survival Distribution')
plt.show()

survival_by_sex = train_data.groupby('Sex')['Survived'].mean()
print("Survival rate by gender:")
print(survival_by_sex)

plt.figure(figsize=(8, 6))
sns.barplot(x=train_data['Sex'], y=train_data['Survived'])
plt.title('Survival Rate by Gender')
plt.show()

survival_by_class = train_data.groupby('Pclass')['Survived'].mean()
print("Survival rate by passenger class:")
print(survival_by_class)

plt.figure(figsize=(8, 6))
sns.barplot(x=train_data['Pclass'], y=train_data['Survived'])
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack')
plt.title('Age Distribution by Survival')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv(r'C:\Users\jidub\OneDrive\Documents\internship\skillcraft\titanic\train.csv')

# Handle missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to high number of missing values
train_data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Convert categorical variables to numeric
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
survival_by_family_size = train_data.groupby('FamilySize')['Survived'].mean()
print("Survival rate by family size:")
print(survival_by_family_size)

plt.figure(figsize=(10, 6))
sns.barplot(x=train_data['FamilySize'], y=train_data['Survived'])
plt.title('Survival Rate by Family Size')
plt.show()