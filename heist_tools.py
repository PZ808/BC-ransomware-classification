import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from tqdm.notebook import tqdm
import pickle
import datetime
import warnings

from scipy import stats
from scipy.stats import boxcox, yeojohnson
from scipy.stats import shapiro

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, plot_roc_curve, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE

RAND_STATE  = 42
READ_STATE = False

def gen_n_day_col(df):
    """
    converts year+day into an integer counting days starting from the first day (time=1) and creates a new column
    :param df: data frame we add a time step col to
    :return: data frame with a new column named t_step_name "at the front" counting days from the start of the data
    """

    t_step_name = 'n_day'
    start = 2011
    year  = 365

    cols = [t_step_name]+list(df.columns) # we want to put the time id at the front of the data frame
    df[t_step_name] = ''

    for idx, _ in tqdm(df.iterrows(),total=df.shape[0]):
        off_set = (df['year'][idx]-start)*year
        df[t_step_name][idx] = df['day'][idx] + off_set

    return df[cols]



def window(df,t0,t_offset):
    """ selects the data from time t0-240 to time t0
    :param df: data frame
    :param t0 persent time
    :param t_offset:   start of the time window
    :return: windowed data frame
    """

    window_bool = (df['n_day']>=(t0-t_offset)) & (df['n_day']<=t0)

    window_ids = list(window_bool[window_bool==True].index)
    #print(window_ids)

    return df.loc[window_ids]

def test_train_time(X, y, t0):
    """ takes the X,y data and prepares test/train sets for 4 time windows
    :param X: input data
    :param y: labels
    :param t0: current time
    :return:
    """
    lags = [480,240,120,60,0]

    X_train = {}
    X_test  = {}
    y_train = {}
    y_test  = {}

    t_first = t0 - lags[0]
    t_min_train = t_first
    train_ids = []
    test_ids = []

    for i in range(len(lags)-1):
        # we break up each lag segment into a 80%/20% train/test split
        #
        # t_first                                       t_last
        # [--------------------------------|------------]
        # ^                               ^ ^          ^
        #  t_min_train         t_max_train  t_min_test  t_max_text
        t_last      = t0 - lags[i+1] # end of time window
        t_max_train = t_first + 0.80 * (t_last-t_first)  # end of train window
        t_min_test  = t_max_train + 1 # start of test window
        t_max_test  = t_last - 1 # end of test window

        train_filter = (X['n_day'] >= t_min_train) & (X['n_day'] <= t_max_train) # get indices
        test_filter  = (X['n_day'] >= t_min_test)  & (X['n_day'] <= t_max_test) # get indices

        X_train[i] = X[train_filter]
        X_test[i]  = X[test_filter]

        y_train[i] = y[train_filter]
        y_test[i]  = y[test_filter]

        train_ids.append(X_train[i].index)
        test_ids.append(X_test[i].index)

    out_ids = [(train_ids[i],test_ids[i]) for i in range(len(lags)-1)]


    return X_train, X_test, y_train, y_test, out_ids

def get_day_number(first_day_of_year, day):
    # day number (0 for monday, 6 for sunday) for day
    return (first_day_of_year + day - 1) % 7

def day_of_week(df):
    """
    :param df: data frame
    :return: a new column with the day of the week
    """
    ''

    day_of_week = np.array([])
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # For each data-point, we consider its year and day
        # and calculate what day of the week (Monday, Tuesday, etc)
        # it was when the transaction happened
        # datetime.datetime(year, 1, 1).weekday() returns the first day of the year
        # using this information, we can calculate the present day

        year_i, day_i = row['year'], row['day']
        day_of_week = np.append (day_of_week, \
                                      get_day_number(datetime.datetime(year_i, 1, 1).weekday(), day_i))
    df['day_of_week'] = day_of_week

    return df

pt = PowerTransformer(method='box-cox')  # transformer to reduce skew, not standard


def plot_box(col_var):
    return sns.boxplot(x=col_var)


def plot_pdf(col_var, title=''):
    # plots distributions

    #self.col = s.name
    sns.distplot(col_var, color="c")
    plt.title(title + ' distribution')
    plt.ylabel('$\pho')
    plt.xlabel(col_var.name)
    plt.plot()


class Feature:
    def __init__(self, df, colname):
        self.df = df
        self.col = colname
        self.vec = np.array(self.df[self.col])
        self.series = self.df[self.col]
        if self.series.dtype == 'O':
            pass
        else:
            self.skew = self.series.skew()
            self.mean = self.series.mean()
            self.var = self.series.var()
            self.std = self.series.std()

    def plot_pdf(self):
        # plots distributions
        sns.distplot(self.series, color="c")
        plt.title(str(self.col) + ' distribution')
        plt.ylabel('density')
        plt.xlabel(self.col)
        plt.plot()

    def plot_box(self):
        return sns.boxplot(x=self.series)

    def apply_pt_for_plots(self):
        return pt.fit_transform(self.vec.reshape(-1, 1))

    def bc(self):
        # boxcox on column
        return boxcox(self.series)[0]

    def yeo(self):
        # yeo on column
        return yeojohnson(self.series)[0]

    def sigmoid(self):
        # logistic function
        return 1 / (1 + np.exp(-self.vec))




def transform(X):
    """
    :param X:
    :return: transformed data
    """

    #X_train_d['address']=train_features['address'].vec
    # cols to standardize (0 mean, unit variance)
    col_names_standardize = ['income',
                             'length',
                             'weight',
                             'neighbors',
                             'looped',
                             'count']
    # cols to encode
    col_names_encode = ['day_of_week']
    col_names_pass = ['year', 'address', 'day']
    # transform the data for use in distance based modeling algorithms [logistic regression]
    transformer = ColumnTransformer(
        [('standardize', StandardScaler(), col_names_standardize),
         ('encode', OneHotEncoder(dtype=int), col_names_encode)],
        remainder='passthrough',
        verbose_feature_names_out=False)

    transformer.fit(X)
    # std. transformed and encoded variables (np arrays)
    Xt = transformer.transform(X)
    # std. transformed and encoded variables (df with verbose headers)

    Xt = pd.DataFrame(Xt, columns=transformer.get_feature_names_out())


    return  Xt





### 4.1.0 Downsampling
class Sampler:
    """
    Class for resampling data
    """

    def __init__(self, X):
        self.X_df = pd.DataFrame(X)
        self.positive_df = pd.DataFrame([], columns=self.X_df.columns)
        self.negative_df = pd.DataFrame([], columns=self.X_df.columns)
        self.combined_df = pd.DataFrame([], columns=self.X_df.columns)
        self.X_train_sm = pd.DataFrame([], columns=self.X_df.columns)
        self.y_train_sm = pd.DataFrame([], columns=['label'])
        self.X_train_rus = pd.DataFrame([], columns=self.X_df.columns)
        self.y_train_rus = pd.DataFrame([], columns=['label'])

    def up_samp_smote(self, X_train, y_train, ratio):
        """Upsamples minority class using SMOTE.
        Ratio argument is the percentage of the upsampled minority class in relation
        to the majority class.
        """
        sm = SMOTE(random_state=RAND_STATE, sampling_strategy=ratio)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        #print(len(X_train_sm), len(y_train_sm))
        self.X_train_sm, self.y_train_sm = X_train_sm, y_train_sm

    def down_samp_rand(self, X_train, y_train, ratio):
        """Downsamples majority class using random sampling.
        Ratio argument is the ratio of minority class to the downsampled majority
        """
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=RAND_STATE)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        self.X_train_rus, self.y_train_rus = X_train_rus, y_train_rus

    def down_samp(self, y_train, prob=0.015):
        index = 0

        for index_df, row in tqdm(self.X_df.iterrows(), total=self.X_df.shape[0]):
            if y_train[index] == 0:
                if np.random.uniform(0, 1) < prob:
                    # picking negative class with a probability=prob
                    self.negative_df = self.negative_df.append(row, ignore_index=True)
                    index += 1
                else:
                    index += 1
            elif y_train[index] == 1:
                # picking all positive data-points
                self.positive_df = self.positive_df.append(row, ignore_index=True)
                index += 1
        self.combined_df = self.positive_df.append(self.negative_df, ignore_index=True)

def plot_roc(model, X, y, plot_title):
    plot_roc_curve(model, X, y)
    plt.title(plot_title)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), label='RandomModel (AUC = 0.50)')
    plt.legend()
    plt.show()


def confusion_mat(y, y_pred):
    cf_matrix = confusion_matrix(y, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    label = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    label = np.asarray(label).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=label, fmt='', cmap='Reds')


def tabulate_scores_df(model_name, y_test, y_pred):
    df2 = pd.DataFrame([[
        model_name,
        accuracy_score(y_true=y_test, y_pred=y_pred),
        precision_score(y_true=y_test, y_pred=y_pred),
        recall_score(y_true=y_test, y_pred=y_pred),
        f1_score(y_true=y_test, y_pred=y_pred)]],
        columns=['Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    return df2

#%%
class Model:
    """

    """
    def __init__(self, x_tr, y_tr, X_tr, X_tst, routine, params, classifier, crss_vldtr=None):
        '''Model'''
        self.classifier = classifier
        self.X_tst = X_tst


        #  randomized search on hyper parameters
        search = RandomizedSearchCV(
            estimator = routine,
            param_distributions = params,
            scoring = 'recall',
            cv = crss_vldtr,
            n_jobs = -1,
            verbose = 1,
            random_state = RAND_STATE,
            return_train_score = True,
            n_iter = 5
        )
        if READ_STATE == True:
            tag = input('Enter head of parameter file to read')
            rf_hps_fname = tag + '_bestparams_sm_e_05.csv'
            print('reading file ', rf_hps_fname)
            bps = pd.read_csv(rf_hps_fname, index_col=False).to_dict('list')
            bps.pop('Unnamed: 0')
            ks = [k for k in bps.keys()]
            vs = [v[0] for v in bps.values()]
            self.best_params = dict( zip(ks,vs) )
        else:
            search.fit(x_tr, y_tr)
            self.best_params = search.best_params_

        print(self.best_params)
        tuned = classifier(**self.best_params,# n_jobs = -1,
                            random_state = RAND_STATE)
        self.tuned = tuned
        # fit
        tuned.fit(x_tr, y_tr)
        # predict
        self.preds_train = tuned.predict(X_tr)
        self.preds_test  = tuned.predict(X_tst)
        self.params      = tuned.get_params()

    def score(self,y_tr,y_tst):
        trn_score_df = tabulate_scores_df('trn', y_tr, self.preds_train)
        tst_score_df = tabulate_scores_df('tst', y_tst, self.preds_test)
        score_df = trn_score_df.append(tst_score_df)
        return score_df

    def roc(self, y_tst, title):
        plot_roc(self.tuned, self.X_tst,y_tst, title)

    def conf_mat(self, y_tr, y_tst):
        plt.figure(figsize=(15,7))

        plt.subplot(1,2,1) # first heatmap
        confusion_mat(y_tr, self.preds_train)

        plt.subplot(1,2,2) # first heatmap
        confusion_mat(y_tst, self.preds_test)

        plt.show()

    def feature_imps(self,colnames):
        feat_ranks_df = pd.DataFrame(np.ravel(self.tuned.feature_importances_), index=colnames)
        feat_ranks_df.columns = ['feature importance']
        feat_ranks_df.sort_values(by='feature importance', ascending=False)
        return feat_ranks_df