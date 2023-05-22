import pandas as pd
import numpy as np
import panphon
from urllib import request
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


def vec_norm(vec_list):
    new_list = []
    for v in vec_list:
        v_n = [0 if item == -1 else item for item in v]
        new_list.append(v_n)
    return new_list


def vec_mean(vec_list):
    df = pd.DataFrame(vec_list)
    vec_mean = df.mean(axis=0).to_list()
    if len(vec_mean) == 0:
        vec_mean = [0] * 24
    return vec_mean


def rescale(xmin, xmax, data):
    data_scaled = np.interp(data, (xmin, xmax), (-1, 1))
    return data_scaled


ft = panphon.FeatureTable()

# Load iconicity dataset
url = "https://raw.githubusercontent.com/jotadwright/NLP_EX3/main/iconicity_ratings.csv"
df = pd.read_csv(request.urlopen(url))

# Scale iconicity values
df['Iconicity'] = df['Iconicity'].apply(lambda x: rescale(-5, 5, x))
df = df.sort_values(by='Iconicity', ascending=False, ignore_index=True)

# Get and normalise phonetic vectors
df['phon_vec'] = df['Word'].apply(lambda x: ft.word_to_vector_list(x, numeric=True))
df['phon_vec'] = df['phon_vec'].apply(vec_norm)

# Get mean of phonetic vectors for each word
df['phon_vec_mean'] = df['phon_vec'].apply(vec_mean)

# List phonetic features and choose which to keep
phon_features = ['1_syl', '2_son', '3_cons', '4_cont', '5_delrel', '6_lat', '7_nas', '8_strid', '9_voi', '10_sg', '11_cg', '12_ant',
                 '13_cor', '14_distr', '15_lab', '16_hi', '17_lo', '18_back', '19_round', '20_velaric', '21_tense', '22_long', '23_hitone', '24_hireg']

keep = ['1_syl', '2_son', '3_cons', '4_cont', '6_lat', '7_nas', '8_strid', '9_voi', '12_ant',
                 '13_cor', '14_distr', '15_lab', '16_hi', '17_lo', '18_back', '19_round', '21_tense']

spl = pd.DataFrame(df['phon_vec_mean'].to_list(), columns=phon_features)
df = df.join(spl)
df = df.drop(['phon_vec', 'phon_vec_mean'], axis=1)

df.to_csv('icon_phon.csv', index=False)


X = df[keep]
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



ft_mean = df.mean(numeric_only=True, axis=0)

print(f'Coefficients')
for i,j in zip(keep,lm.coef_):
    print(i,j)

print('\nMean feature values', ft_mean)

print('\n', 'Test set metrics')
print('R squared: {:.2f}'.format(lm.score(X, y) * 100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
