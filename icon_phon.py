import pandas as pd
import numpy as np
import panphon
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics


def flatten_list(_2d_list):
    """Takes a 2D list and flattens to 1D"""
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def vec_norm(vec_list, max_len):

    empty_vec = [0] * 24
    while len(vec_list) < max_len:
        vec_list.append(empty_vec)

    return vec_list

def vec_mean(vec_list):
    df = pd.DataFrame(vec_list)
    vec_mean = df.mean(axis=0).to_list()
    if len(vec_mean) == 0:
        vec_mean = [0] * 24
    return vec_mean

ft = panphon.FeatureTable()


df = pd.read_csv('iconicity_senses-master/data/iconicity_ratings.csv')

# Scale iconicity values
scaler = MinMaxScaler((-1,1))
df['Iconicity'] = scaler.fit_transform(df[['Iconicity']])
df = df.sort_values(by='Iconicity', ascending=False, ignore_index=True)


df['phon_vec'] = df['Word'].apply(lambda x: ft.word_to_vector_list(x, numeric=True))
max_len = max(df['phon_vec'].apply(len))

df['phon_vec_mean'] = df['phon_vec'].apply(vec_mean)
# print("MAX LEN MEAN")
# print(max(df['phon_vec_mean'].apply(len)))
# print("MIN LEN MEAN")
# print(min(df['phon_vec_mean'].apply(len)))

df['phon_vec'] = df['phon_vec'].apply(lambda x: vec_norm(x, max_len))
df['phon_vec'] = df['phon_vec'].apply(flatten_list)


print(type(df['phon_vec_mean'][0]))

X = df['phon_vec_mean'].to_list()
y = df['Iconicity']

# Split data in 70/30 test/train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

print(X_train[0])

# Load and fit the linear regression model to the training set.
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

# Get 5-fold cross-validation scores for the training set
folds = KFold(n_splits=5, shuffle=True, random_state=33)
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=folds)

print('K-fold CV R squared:', scores, "\n")

# Get predictions and print a sample
y_pred = lm.predict(X_test)

print("Sample of predictions")
lm_diff = pd.DataFrame({'iconicity': y_test, 'lm_pred': y_pred})
lm_diff = lm_diff.join(df['Word']).sort_index()
print(lm_diff.sample(20).sort_values(by='iconicity', ascending=False, ignore_index=True), "\n")

# Get and print LM test set metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Test set metrics')
print('R squared: {:.2f}'.format(lm.score(X, y) * 100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
