# Continuacao do projeto PIBITI, agora utilizando mlp
# imports ==============================================================================================================

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
import theano
import theano.tensor as T
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from cnn import evaluate_lenet5
from mlpstudent import test_mlp

# ======================================================================================================================
# Read the data
students_data = pd.read_csv('student-por.csv', sep=';', true_values=['yes'], false_values=['no'])


# Add boolean column PassFail
def passfail(row):
    if row['G3'] >= 10:
        return True
    else:
        return False


students_data['PassFail'] = students_data.apply(lambda row: passfail(row), axis=1)

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

# columns types
students_num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
students_cat_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and
                     X[cname].dtype == "object"]
students_bool_cols = [cname for cname in X.columns if X[cname].dtype == "bool"]


def scale_numeric(data, numeric_columns, scaler):
    for col in numeric_columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# We can now define the scaler we want to use and apply it to our dataset
scaler = StandardScaler()
X = scale_numeric(X, [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']], scaler)

# Keep selected columns only
my_cols = students_cat_cols + students_num_cols + students_bool_cols
X = X[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, students_num_cols),
        ('cat', categorical_transformer, students_cat_cols),
        ('bool', 'passthrough', students_bool_cols)])

X = pd.DataFrame(preprocessor.fit_transform(X))
Y = pd.DataFrame(Y)

# Divide data into training, validation and test subsets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_valid, Y_valid, train_size=0.8, test_size=0.2,
                                                    shuffle=True)
train_set = (X_train, Y_train)
print(f'X_train shape = {X_train.shape}')
print(f'Y_trains shape = {Y_train.shape}')
valid_set = (X_valid, Y_valid)
print(f'X_valid shape = {X_valid.shape}')
print(f'Y_valid shape = {Y_valid.shape}')
test_set = (X_test, Y_test)
print(f'X_test shape = {X_test.shape}')
print(f'Y_test shape = {Y_test.shape}')


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow).flatten()
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


te_set_x, te_set_y = shared_dataset(test_set)
val_set_x, val_set_y = shared_dataset(valid_set)
tr_set_x, tr_set_y = shared_dataset(train_set)
dataset = [(tr_set_x, tr_set_y), (val_set_x, val_set_y), (te_set_x, te_set_y)]

# MLP ==================================================================================================================
if __name__ == '__main__':
    test_mlp(dataset)

# CNN ==================================================================================================================
if __name__ == '__main__':
    evaluate_lenet5(dataset)
