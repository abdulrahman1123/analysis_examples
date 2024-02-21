import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import RocCurveDisplay,classification_report, recall_score,confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import mannwhitneyu as mn_test, ttest_ind as ttest


#sns.set(style="darkgrid")
#sns.set_context("paper")

def choose_model(model, c_type = 'best', plot_result = True):
    """
    This function chooses a model from a set of models identified using LogisticRegressionCV
    It can return the best model (model.C_) or the most parsimonious model, which is the model whose score is
    within 1 standard error from the best score
    :param model: logistic regression model with cross validation (LogisticRegressionCV)
    :param c_type: type of C value to return,
                   can be either 'best' for best model, or 'par' for the most parsimonious model
    :param plot_result: whether to plot the reult and show the best model and parsimonious model on the same figure
    :return: C value for the chosen model
    """
    n_folds = model.coefs_paths_[1.0].shape[0]
    c_vals = model.Cs_
    best_c = model.C_

    best_c_ind = np.where(np.abs(c_vals - model.C_) < 1e-10)[0][0]

    included_vars = np.sum(model.coefs_paths_[1.0].mean(axis=0) != 0,  axis=1) - 1  # the -1 is make sure the intercept is not included
    included_vars = included_vars[
        [int(item) for item in np.linspace(0, len(included_vars) - 1, 30)]]  # Take only 30 samples from included_vars
    scores = model.scores_[1.0].mean(axis=0)
    scores_sem = model.scores_[1.0].std(axis=0) / np.sqrt(n_folds)

    # Get 1 standard error of the mean (SEM) from the best accuracy,
    best_sem = scores_sem[best_c_ind]

    # finds the last point where scores are within one SEM from best score
    c1se_ind = np.where(scores[best_c_ind] - scores[0:best_c_ind] < best_sem)[0][0]
    c1se = model.Cs_[c1se_ind]  # least acceptable score
    if plot_result:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax3 = ax.twiny()
        ax3.set_xticks(np.arange(0, len(included_vars) + 2), [''] + list(included_vars) + [''], fontsize = 12)
        ax3.tick_params(width=0, length = 0)
        ax3.set_xlabel('Included Variables', fontsize = 18)
        ax.axvline(x=np.log(best_c), color='grey', ls='-', lw=1, label='Best Score Model')
        ax.axvline(x=np.log(c1se), color='grey', ls='-.', lw=1, label='Parsimonious Model')
        ax.errorbar(np.log(model.Cs_), scores, scores_sem, fmt='o', linewidth=1,
                    color='grey', mfc='royalblue', mec='none', capsize=4)
        ax.legend()
        x_axis_text = np.round(ax.get_xticks()[1:-1],1)
        y_axis_text = np.round(ax.get_yticks()[1:-1],1)
        ax.set_xticks(ticks =x_axis_text, labels =  x_axis_text, fontsize = 12)
        ax.set_yticks(ticks =y_axis_text, labels =  y_axis_text, fontsize = 12)
        ax.set_xlabel('log(C)', fontsize = 18)
        ax.set_ylabel('Accuracy', fontsize = 18)

    if c_type=='best':
        return model.C_[0]
    elif c_type == 'par':
        return c1se
    else:
        raise Warning("c_type can only be set to 'best' or 'par'")


def model_performance(model,X_test,y_test, col_names):
    """
    Print the coefficients and compute accuracy
    :param model: the model to be tested
    :return: print the coeffcieints and compute accuracy
    """
    # let's have a look at the coefficients and see if anything was removed
    print('Coefficients:')
    coefs = [model.intercept_[0]] + list(model.coef_[0])
    coefs = [str(np.round(item,3)) if item!=0 else "-" for item in coefs]
    coef_names = ['intercept'] + list(col_names)[1:-1]
    coefficients = pd.DataFrame(coefs, index=coef_names,columns=['value'])
    print(coefficients)
    print('\n Scores:')

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return pd.DataFrame([[sens, spec, accuracy]], columns=['Sensitivity', 'Specificity', 'Accuracy'])


def plot_train_test(X_train, X_test, X_val = None):
    """
    Plot the sizes of training and testing datasets for visualization reasons
    :param X_train: Training dataset that contains training subjects as rows
    :param X_test: Testing dataset that contains testing subjects as rows
    :return: plots a stacked horizontal bar plot showing training and testing dataset sizes
    """
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    val_size = 0 if np.all(pd.isnull(X_val)) else X_val.shape[0]

    fig, ax = plt.subplots(figsize=(12.5, 1))
    ax.barh(['Data Split'], train_size, edgecolor='black', color='lightsteelblue')
    ax.barh(['Data Split'], val_size, left=train_size, edgecolor='black', color='grey')
    ax.barh(['Data Split'], test_size, left=train_size+val_size, edgecolor='black', color='white')
    ax.text(2, 0, f'Training sample = {train_size} subjects \n(Training the model)',
            size=14, va='center_baseline')
    if not np.all(pd.isnull(X_val)):
        ax.text(train_size + 1, 0, f'Validation = {val_size} subjects\n(finding best parameters)', 
            size=14, va='center_baseline')
    ax.text(train_size+val_size + 2, 0, f'Testing sample = {test_size} subjects\n(held-out for final testing)',
            size=14, va='center_baseline')
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_xlabel('');
    ax.set_ylabel('')
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.01, right=1, top=0.95, bottom=0.05)
    plt.show()

def calc_ttest(inc_data, ind_var, lvls, dep_var,tick_labels = None, colors = ['blue','red'], plot_result = False, return_ax = False, force_ttest = False, test_type = '2samp', verbose = False, y_label = None):
    '''
    simple function to compute t test or Mann-Whitney test (for data that vilates the normality assumption)
    :param inc_data: the dataframe to be included
    :param ind_var: independent variable
    :param lvls: the two levels fo independent variable that should be compared
    :param dep_var: the dependent variable
    :param verbose: whether to return feedback or error messages
    :return: a data frame with the statsitic (Mann-Whitney U or T-Test T), p-value and test type (MN or T-test)
    '''
    if tick_labels == None:
        tick_labels = lvls

    dep = inc_data[dep_var].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    ind = inc_data[ind_var]

    data1= dep.loc[pd.notna(dep) & (ind==lvls[0])].astype('float64')
    data2= dep.loc[pd.notna(dep) & (ind==lvls[1])].astype('float64')

    if len(data1)<3 or len(data2)<3:
        if verbose:
            print(f'Smallest group has {np.min([data1.shape[0],data2.shape[0]])} cases, returning NA')
        if return_ax:
            return pd.DataFrame([[np.nan]*6], columns=['statistic','p-val','test_type','df','Mean1','Mean2']), [None,None]
        else:
            return pd.DataFrame([[np.nan]*6], columns=['statistic','p-val','test_type','df','Mean1','Mean2'])
    if (shapiro(data1)[1]>0.05 and shapiro(data2)[1]>0.05) or (len(data1)>30 and len(data2)>30) or force_ttest:
        test_res = ttest(data1,data2)
        stat, pval, testtype = np.round(test_res[0],3),np.round(test_res[1],3),'T-test'
    else:
        test_res = mn_test(data1, data2)
        stat, pval, testtype = np.round(test_res[0],3),np.round(test_res[1],3),'MW-test'
    df= data1.shape[0]+data2.shape[0]-2
    m1,m2 = np.round(data1.mean(),2), np.round(data2.mean(),2)
    test_df =pd.DataFrame([[stat, pval, testtype, df,m1,m2]])

    test_df.columns=['statistic','p-val','test_type','df','Mean1','Mean2']


    if return_ax:
        return test_df, (fig, ax)
    else:
        return test_df
