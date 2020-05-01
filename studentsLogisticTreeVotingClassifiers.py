# pibit empresa 'SELEÇÃO DE CARACTERÍSTICAS PARA PREVISÃO DO DESEMPENHO DE ALUNOS EM CURSOS EAD'
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
plt.show()

# Separate target from predictors
Y = students_data.PassFail
X = students_data.drop(['absences', 'G1', 'G2', 'G3', 'PassFail'], axis=1)


def conf_matrix(preds, Y_valid):
    y_valid = [i for i in Y_valid]
    truepred = [0, 0]
    falsepred = [0, 0]
    for i in range(len(preds)):
        if preds[i]:
            if preds[i] == y_valid[i]:
                truepred[0] += 1
            else:
                truepred[1] += 1
        else:
            if preds[i] != y_valid[i]:
                falsepred[0] += 1
            else:
                falsepred[1] += 1
    return pd.DataFrame(np.array([truepred, falsepred]), columns=['True', 'False'], index=['TruePred', 'FalsePred'])


def scale_numeric(data, numeric_columns, scaler):
    for col in numeric_columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# We can now define the scaler we want to use and apply it to our dataset
scaler = StandardScaler()
X = scale_numeric(X, [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']], scaler)

# Divide data into training and validation subsets
X_train_full, X_valid_full, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, stratify=Y,
                                                                shuffle=True)
# Get shape of test and training sets
print('Training Set:')
print('Number of datapoints: ', X_train_full.shape[0])
print('Number of features: ', X_train_full.shape[1])
print('Test Set:')
print('Number of datapoints: ', X_valid_full.shape[0])
print('Number of features: ', X_valid_full.shape[1])

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
cat_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
            X_train_full[cname].dtype == "object"]
# Select numerical columns
num_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# Select boolean columns
bool_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "bool"]

# Keep selected columns only
my_cols = cat_cols + num_cols + bool_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('bool', 'passthrough', bool_cols)])

# Choose model
model1 = DecisionTreeClassifier(criterion='entropy')
model2 = RandomForestClassifier(n_estimators=100)
model3 = LogisticRegression(solver='lbfgs')
voting_clf = VotingClassifier(estimators=[('dt', model1), ('rf', model2), ('lr', model3)],
                              # here we select soft voting, which returns the argmax of the sum of
                              # predicted probabilities
                              voting='soft', weights=[1, 1, 1])
models = [model1, model2, model3, voting_clf]
# Comparing models
for mod in models:
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', mod)])
    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, Y_train)
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)
    # Confusion Matrix
    print("Confusion Matrix:\n", conf_matrix(preds, Y_valid))
    # Evaluate the model without xvalidation
    score = accuracy_score(Y_valid, preds)
    print("Accuracy score without xvalidation:\n", score)
    # Evaluate the model with xvalidation
    scores = cross_val_score(my_pipeline, X_train, Y_train, cv=5, scoring='accuracy')
    print("Accuracy scores:\n", scores)
    print("Average Accuracy score (across experiments):")
    print(scores.mean())
    print('----------------------------------------------------------')