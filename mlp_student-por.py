# Continuacao do projeto PIBITI, agora utilizando CNN

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import theano
import theano.tensor as T

# Read the data
students_data = pd.read_csv('student-por.csv', sep=';', true_values=['yes'], false_values=['no'])


# Add boolean column PassFail
def passfail(row):
    if row['G3'] >= 10:
        return True
    else:
        return False


students_data['PassFail'] = students_data.apply(lambda row: passfail(row), axis=1)
# columns types
students_num_cols = [cname for cname in students_data.columns if students_data[cname].dtype in ['int64', 'float64']]
students_cat_cols = [cname for cname in students_data.columns if students_data[cname].nunique() < 10 and
                     students_data[cname].dtype == "object"]
students_bool_cols = [cname for cname in students_data.columns if students_data[cname].dtype == "bool"]

# data visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
students_data.G3.hist(ax=axes[0])
axes[0].set_title("Distribuição de G3")
sns.countplot(x="PassFail", data=students_data, ax=axes[1])
axes[1].set_title("Distribuição entre Aprovados/Reprovados")
plt.grid(True, axis='y')
# plt.show()

# Separate target from predictors
Y = students_data.PassFail
X = students_data.drop(['absences', 'G1', 'G2', 'G3', 'PassFail'], axis=1)


def scale_numeric(data, numeric_columns, scaler):
    for col in numeric_columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# We can now define the scaler we want to use and apply it to our dataset
scaler = StandardScaler()
X = scale_numeric(X, [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']], scaler)

# Divide data into training, validation and test subsets
X_train_full, X_valid_full, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)
X_valid_full, X_test_full, Y_valid, Y_test = train_test_split(X_valid_full, Y_valid, train_size=0.8, test_size=0.2,
                                                              shuffle=True)
# Get shape of training, validation and test sets
print('Training Set:')
print('Number of datapoints: ', X_train_full.shape[0])
print('Number of features: ', X_train_full.shape[1])
print('Validation Set:')
print('Number of datapoints: ', X_valid_full.shape[0])
print('Number of features: ', X_valid_full.shape[1])
print('Test Set:')
print('Number of datapoints: ', X_test_full.shape[0])
print('Number of features: ', X_test_full.shape[1])


