import pandas as pd
import numpy as np
import urllib
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics

# Load Fasttext pre-trained vectors.
# Available from https://drive.google.com/file/d/1XNb-1GhhUmFnhLrfviXar_89YhnjyefO/view?usp=share_link
print("Loading DS model...")
ds_model = gensim.models.KeyedVectors.load_word2vec_format('cc.en.150.vec.gz', binary=False)
print("DS model loaded.", "\n")
words_covered = list(ds_model.index_to_key)


def sem_vec(word):
    """Gets vector for word, gives empty vector for out of vocabulary words"""
    vec_len = len(ds_model.get_vector('cat'))

    if ds_model.has_index_for(word):
        vec = ds_model.get_vector(word)
    else:
        vec = [0] * vec_len

    return vec


def reduce_dim(vector, n):
    """Reduces dimensionality of a vector to n via PCA"""
    pca = PCA(n_components=n)
    nd_vec = pca.fit_transform(vector)
    return nd_vec


# Load iconicity dataset
url = "https://raw.githubusercontent.com/jotadwright/NLP_EX3/main/iconicity_ratings.csv"
df = pd.read_csv(urllib.request.urlopen(url))

# Scale iconicity values
scaler = MinMaxScaler((-1,1))
df['Iconicity'] = scaler.fit_transform(df[['Iconicity']])
df = df.sort_values(by='Iconicity', ascending=False, ignore_index=True)

# Get semantic vectors for the vocabulary
df['sem_vec'] = df['Word'].apply(sem_vec)

# Reduce dimensionality of word embeddings
sem_vec_red = reduce_dim(df['sem_vec'].to_list(), 20)

df['sem_vec_red'] = sem_vec_red.tolist()

# Set semantic vector dimensions as independent variables and iconicity rating
# as dependent variable for linear regression
X = df['sem_vec_red'].to_list()
y = df['Iconicity']

# Split data in 70/30 test/train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

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

int_only = [lm.intercept_] * len(y)

print('Intercept:', lm.intercept_)
print('Intercept R squared: {:.2f}'.format(lm.score(X, int_only) * 100))




lm_diff.to_csv('lm_test.csv', index=False)