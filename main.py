import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
PassengerId - Passenger number
Survived - 0 = Dead 1 = Alive
Pclass - 1 = First class 2 = Second class 3 = Third class
Name - Name of passenger
Sex - Gender
Age - Age of passenger
SibSp - Number of siblings/Spouses
Parch - Number of Parents/Children Aboard - 
Ticket - Ticket
Fare - Price of ticket(In British pound)
Cabin - Location of stay for passenger.
Embarked - C = Cherbourg Q = Queenstown S = Southampton

"""

# TODO Clean dataset
# - Cabin has 78.2% of the data missing
# - Name in the correct format.
# - Restructuring age.
# - Understand Fare price (British Pounds)
# - Understand Ticket data.


train_data = pd.read_csv("tested.csv")
missing_data = train_data.isnull().sum()
missing_data = missing_data[missing_data > 0]

missing_data_percentage = (missing_data / len(train_data)) * 100
missing_data_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage (%)': missing_data_percentage})
missing_data_info.sort_values(by='Percentage (%)', ascending=False)

# print(missing_data_info)


train_data['CabinKnown'] = train_data['Cabin'].notnull().astype(int)

# Filling missing 'Age' values with the median age
median_age = train_data['Age'].median()
train_data['Age'].fillna(median_age, inplace=True)

# Filling missing 'Embarked' values with the mode (most common value)
mode_embarked = train_data['Embarked'].mode()[0]
train_data['Embarked'].fillna(mode_embarked, inplace=True)

# Rechecking for missing data
updated_missing_data = train_data.isnull().sum()
updated_missing_data = updated_missing_data[updated_missing_data > 0]

updated_missing_data_percentage = (updated_missing_data / len(train_data)) * 100

# Updated missing data information
updated_missing_data_info = pd.DataFrame(
    {'Remaining Missing Values': updated_missing_data, 'Percentage (%)': updated_missing_data_percentage})
updated_missing_data_info.sort_values(by='Percentage (%)', ascending=False)

# print(updated_missing_data_info)


summary_statistics = train_data.describe(include='all')
summary_statistics.transpose()
print(summary_statistics)


# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Categorical: Distribution of passengers by sex
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', data=train_data)
plt.title('Distribution of Passengers by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Hierarchical: Survival rates by class and sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_data)
plt.title('Survival Rates by Class and Sex')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Relational: Fare paid vs. Age
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', style='Pclass', data=train_data)
plt.title('Fare vs. Age (Colored by Survival, Style by Class)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Temporal: Histogram of Ages
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'], bins=30, kde=True)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()