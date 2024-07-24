import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS
import torch
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns

########################################
# DATA FUNCTIONS #######################
########################################

def get_data(df, order=1, size=0.3, sinusoidal=False, combined = False):
    """
    Function to get the data for the regression in the right formats.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe containing the features and the output variable
        order (int, optional): Order of polynomial features. Defaults to 1.
        size (float, optional): Test size. Defaults to 0.3.
        sinusoidal (bool, optional): Whether to include sinusoidal features. Defaults to False.
        combined (bool, optional): Whether to include both polynomial and sinusoidal features. No need to specificy sinusoidal. Detaults to False.
        
    Returns:
        tuple: containing train and test sets. 
    """
    X = np.array(df[["orientation", "phase", "spatial_frequency"]])
    y = np.array(df["spike_count"])
    bias = np.ones((X.shape[0], 1))

    if sinusoidal:
        X_sin = np.apply_along_axis(lambda x: np.sin(x), 0, X)
        X_cos = np.apply_along_axis(lambda x: np.cos(x), 0, X)
        X_feat = np.hstack((X_sin, X_cos))
    elif combined:
        X_sin = np.apply_along_axis(lambda x: np.sin(x), 0, X)
        X_cos = np.apply_along_axis(lambda x: np.cos(x), 0, X)
        X_poly = preprocessing.PolynomialFeatures(order, include_bias=False).fit_transform(X)
        X_feat = np.hstack((X_poly, X_sin, X_cos))
    else:
        X_feat = preprocessing.PolynomialFeatures(order, include_bias=False).fit_transform(X)
    
    X_feat_norm = preprocessing.StandardScaler().fit_transform(X_feat)
    X_final = np.hstack((bias, X_feat_norm))
        
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=size, random_state=42)

    return X_train, X_test, y_train, y_test

def get_data_for_MLP(X_train, X_test, y_train, y_test):
    """
    Assuming inputs are from get_data, the function returns the Tensor datasets and dataloaders

    Args:
        X_train (numpy.ndarray)
        X_test (numpy.ndarray)
        y_train (numpy.ndarray)
        y_test (numpy.ndarray)

    Returns:
        tuple: datasets and dataloader for train and test respectively
    """

    torch.manual_seed(42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)

    return train_dataset, train_dataloader, test_dataset, test_dataloader

########################################
# LINEAR REGRESSION FUNCTIONS ##########
########################################

def backward_elimination(y, X, sign_level=0.1):
  """
  Model selection with step-down approach, removing all values non significant values iteratively.

  Args:
      y (numpy.ndarray): train y values
      X (numpy.ndarray): full train matrix with variables of the model
      sign_level (float, optional): significance level. Defaults to 0.1.

  Returns:
      list: indices of the input feature that are statistically significant
  """
  current_features = list([i for i in range(len(X[0]))])  # initialize with all indices
  while True:
    least_sig_pval = -1
    least_sig_index = None

    for i in range(len(current_features)):
      model = OLS(y, X[:, current_features]).fit()
      pval = model.pvalues[i]

      if pval > least_sig_pval:
        least_sig_pval = pval
        least_sig_index = i

    if least_sig_pval < sign_level: # check if stopping criterion is met
      break
    else:
      current_features.remove(current_features[least_sig_index]) # remove the least significant variable index

  return current_features

def get_result(model, X_test, y_test):
    """
    Get summary results of RMSE of the model

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): regression model
        input (numpy.ndarray): test matrix with variables of the model
        test (numpy.ndarray): test y values
    """

    print(f"R Squared: {model.rsquared:.3f}, Adjusted R Squared {model.rsquared_adj:.3f}") #r2 and adjusted r2
    # print(f"RMSE of the Model: {np.sqrt(model.mse_model):3f}") # explained sum of squares divided by the model degrees of freedom
    # print(f"RMSE of the Residuals: {np.sqrt(model.mse_resid):3f}") # sum of squared residuals divided by the residual degrees of freedom
    print(f"Train RMSE: {np.sqrt(model.mse_total):.3f}") # uncentered total sum of squares divided by the number of observations

    coeff = 1 / len(y_test)
    error = np.sum((model.predict(X_test) - y_test)**2)
    print(f"Test RMSE: {np.sqrt(coeff * error):.3f}") # test loss
    
def get_prediction(df, model, orientation, spatial_frequency, phase, variables_to_keep, order=1, size=0.3, sinusoidal=False, combined = False):
    """
    Function to get the prediction for the regression in the right formats.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe containing the features and the output variable, that are used for the scaling
        model (OLS object): model that we will use for the regression
        orientation(float): orientation of the grating
        spatial_frequency(float): spatial frequency of the grating
        phase(float): phase of the grating
        variables_to_keep(list): variables of the regression that we want to keep
        order (int, optional): Order of polynomial features. Defaults to 1.
        size (float, optional): Test size. Defaults to 0.3.
        sinusoidal (bool, optional): Whether to include sinusoidal features. Defaults to False.
        combined (bool, optional): Whether to include both polynomial and sinusoidal features. No need to specificy sinusoidal. Detaults to False.

    Returns:
        tuple: prediction and lower/upper confidence values
    """

    #useful for scaling
    X1 = np.array(df[["orientation", "phase", "spatial_frequency"]])

    if sinusoidal:
        X_sin1 = np.apply_along_axis(lambda x: np.sin(x), 0, X1)
        X_cos1 = np.apply_along_axis(lambda x: np.cos(x), 0, X1)
        X_feat1 = np.hstack((X_sin1, X_cos1))
    elif combined:
        X_sin1 = np.apply_along_axis(lambda x: np.sin(x), 0, X1)
        X_cos1 = np.apply_along_axis(lambda x: np.cos(x), 0, X1)
        poly_features = preprocessing.PolynomialFeatures(order, include_bias=False)
        X_poly1 = poly_features.fit_transform(X1)
        X_feat1 = np.hstack((X_poly1, X_sin1, X_cos1))
    else:
        poly_features = preprocessing.PolynomialFeatures(order, include_bias=False)
        X_feat1 = poly_features.fit_transform(X1)

    scaler = preprocessing.StandardScaler()
    X_feat_norm = scaler.fit_transform(X_feat1)

    #actual prediction
    X = np.array([[orientation, phase, spatial_frequency]])
    bias = np.ones((X.shape[0], 1))
    variables_to_keep = np.array(variables_to_keep)

    if sinusoidal:
        X_sin = np.apply_along_axis(lambda x: np.sin(x), 0, X)
        X_cos = np.apply_along_axis(lambda x: np.cos(x), 0, X)
        X_feat = np.hstack((X_sin, X_cos))
    elif combined:
        X_sin = np.apply_along_axis(lambda x: np.sin(x), 0, X)
        X_cos = np.apply_along_axis(lambda x: np.cos(x), 0, X)
        X_poly = poly_features.transform(X)
        X_feat = np.hstack((X_poly, X_sin, X_cos))
    else:
        X_feat = poly_features.transform(X)

    X_feat_norm = scaler.transform(X_feat)
    X_final = np.hstack((bias, X_feat_norm))

    prediction = model.predict(X_final[:, variables_to_keep][0])[0]
    lower_bound = model.get_prediction(X_final[:, variables_to_keep]).summary_frame()["obs_ci_lower"][0]
    upper_bound = model.get_prediction(X_final[:, variables_to_keep]).summary_frame()["obs_ci_upper"][0]

    return prediction, lower_bound, upper_bound

def get_equation(final_features, model, sinusoidal=False):
    """
    Returns the estimated equation as a string from the fitted OLS object.
    
    Parameters:
    final_features (list): A list of column indices to include in the equation.
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted OLS object.
    sinusoidal (bool, optional): If True, column names are assumed to be trigonometric functions of the original variables. Detaults to False.
    
    Returns:
    str: The estimated equation as a string.
    """
    
    coefs = model.params
    equation = f"y = {coefs[0]:.2f}"
    
    if sinusoidal:
        column_names = ['const', 'sin(orient)', 'sin(phase)', 'sin(freq)',
                        'cos(orient)', 'cos(phase)', 'cos(freq)']
    else:
        column_names = ['const', 'orient', 'phase', 'freq',
                        'orient^2', '(orient * phase)', '(orient * freq)',
                        'phase^2', '(phase * freq)', 'freq^2',
                        'sin(orient)', 'sin(phase)', 'sin(freq)',
                        'cos(orient)', 'cos(phase)', 'cos(freq)']
    
    for i, feat in enumerate(final_features[1:], start=1):
        coef = coefs[i]
        term = f" + {coef:.2f} * {column_names[feat]}"
        equation += term
    
    return f"Final equation --> {equation}"

########################################
# PLOTTING FUNCTIONS ###################
########################################

def plot_data(dfs):
    """
    Plots spike count per each variable, for each neuron in dfs

    Args:
        dfs (dict): containts all pandas.core.frame.DataFrame objects for each neuron
    """
    for i in range(5):
        data = dfs[i]
        print(f"Neuron {i+1}, Number of Samples: {data.shape[0]}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle(f'Neuron {i+1} Spike Count per Variable', fontsize=10)

        for j, col in enumerate(["orientation", "phase", "spatial_frequency"]):
            axes[j].scatter(data[col], data["spike_count"])
            axes[j].set_xlabel(col)
            axes[j].grid(True)
        plt.show()


def _plot_residuals(model):
    """
    Plot residuals of the linear regression model

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): regression model
    """
    plt.figure(figsize=(5, 4))
    sns.histplot(model.resid, kde=True, color="blue", bins=10)
    plt.xlabel("Residuals", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Histogram of Residuals", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_regression_2D(df, model, var_to_plot1, var_to_plot2, var_to_keep, order = 1, size = 0.3, sinusoidal = False, combined = False, orientation = 0, spatial_frequency = 0, phase = 0):
    # %matplotlib widget

    resolution = 50
    x = np.linspace(min(df[var_to_plot1]) - 0.1, max(df[var_to_plot1]) + 0.1, resolution)
    y = np.linspace(min(df[var_to_plot2]) - 0.1, max(df[var_to_plot2]) + 0.1, resolution)

    z = np.ones((1,len(x)))

    #get the predictions
    if var_to_plot1 == 'spatial_frequency':
        if var_to_plot2 == 'phase':
            for i in y:
                temp = []
                for j in x:
                    temp.append(get_prediction(df, model, orientation, j, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                z = np.vstack((z, temp))
        if var_to_plot2 == 'orientation':
            for i in y:
                temp = []
                for j in x:
                    temp.append(get_prediction(df, model, i, j, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                z = np.vstack((z, temp))
    if var_to_plot1 == 'phase':
        if var_to_plot2 == 'spatial_frequency':
            for i in y:
                temp = []
                for j in x:
                    temp.append(get_prediction(df, model, orientation, i, j, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                z = np.vstack((z, temp))
        if var_to_plot2 == 'orientation':
            for i in y:
                temp = []
                for j in x:
                    temp.append(get_prediction(df, model, i, spatial_frequency, j, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                z = np.vstack((z, temp))
    if var_to_plot1 == 'orientation':
        if var_to_plot2 == 'phase':
            for i in y:
                temp = []
                for j in x:
                    temp.append(get_prediction(df, model, j, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                z = np.vstack((z, temp))
        if var_to_plot2 == 'spatial_frequency':
            for i in y:
                temp = []
                for j in x:
                    temp.append(get_prediction(df, model, j, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                z = np.vstack((z, temp))

    z = z[1:,:]

    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)

    ax.scatter(df[var_to_plot1], df[var_to_plot2], df['spike_count'])

    ax.set_xlabel(var_to_plot1)
    ax.set_ylabel(var_to_plot2)
    ax.set_zlabel('spike count')
    ax.set_title('3D Surface Plot')

    plt.show()

def plot_regression_1D(df, model, var_to_plot, var_to_keep, order = 1, size = 0.3, sinusoidal = False, combined = False, orientation = 0, spatial_frequency = 0, phase = 0):
    plt.clf()
    # %matplotlib inline

    fig, ax = plt.subplots(figsize=(7, 5))

    if var_to_plot == 'spatial_frequency':
        x = np.linspace(min(df[var_to_plot]) - 0.01, max(df[var_to_plot]) + 0.01, 50)
    else:
        x = np.linspace(min(df[var_to_plot]) - 0.1, max(df[var_to_plot]) + 0.1, 50)

    y = []
    iv_l = []
    iv_u = []
    if var_to_plot == 'spatial_frequency':
        for i in x:
            y.append(get_prediction(df, model, orientation, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
            iv_l.append(get_prediction(df, model, orientation, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[1])
            iv_u.append(get_prediction(df, model, orientation, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[2])
    if var_to_plot == 'orientation':
        for i in x:
            y.append(get_prediction(df, model, i, spatial_frequency, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
            iv_l.append(get_prediction(df, model, i, spatial_frequency, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[1])
            iv_u.append(get_prediction(df, model, i, spatial_frequency, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[2])
    if var_to_plot == 'phase':
        for i in x:
            y.append(get_prediction(df, model, orientation, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
            iv_l.append(get_prediction(df, model, orientation, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[1])
            iv_u.append(get_prediction(df, model, orientation, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[2])


    ax.plot(x, y, "r--.", linewidth=2, label="MLP")
    # ax.plot(x, iv_u, "g--", linewidth=0.5, label = "CI")
    ax.plot(x, iv_l, "g--", linewidth=0.5)

    #scatter points
    x = df[var_to_plot]
    y = df['spike_count']
    ax.plot(x, y, "o", label="data")

    ax.set_xlabel(var_to_plot, fontsize=12)
    ax.set_ylabel("Spike Count", fontsize=12)
    ax.set_title("Fitted Model by MLP", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)

    plt.show()

def plot_regression_1D_all(df, model, var_to_keep, order = 1, size = 0.3, sinusoidal = False, combined = False, orientation = 0, spatial_frequency = 0, phase = 0):
    plt.clf()
    # %matplotlib inline

    fig, ax = plt.subplots(1,3,figsize=(21, 5))
    count = 0
    for var_to_plot in ['orientation', 'spatial_frequency', 'phase']:
        if var_to_plot == 'spatial_frequency':
            x = np.linspace(min(df[var_to_plot]) - 0.01, max(df[var_to_plot]) + 0.01, 50)
        else:
            x = np.linspace(min(df[var_to_plot]) - 0.1, max(df[var_to_plot]) + 0.1, 50)

        y = []
        iv_l = []
        iv_u = []
        if var_to_plot == 'spatial_frequency':
            for i in x:
                y.append(get_prediction(df, model, orientation, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                iv_l.append(get_prediction(df, model, orientation, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[1])
                iv_u.append(get_prediction(df, model, orientation, i, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[2])
        if var_to_plot == 'orientation':
            for i in x:
                y.append(get_prediction(df, model, i, spatial_frequency, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                iv_l.append(get_prediction(df, model, i, spatial_frequency, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[1])
                iv_u.append(get_prediction(df, model, i, spatial_frequency, phase, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[2])
        if var_to_plot == 'phase':
            for i in x:
                y.append(get_prediction(df, model, orientation, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[0])
                iv_l.append(get_prediction(df, model, orientation, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[1])
                iv_u.append(get_prediction(df, model, orientation, spatial_frequency, i, var_to_keep, order = order, size = size, sinusoidal=sinusoidal, combined = combined)[2])


        ax[count].plot(x, y, "r--.", linewidth=2, label="OLS")
        ax[count].plot(x, iv_u, "g--", linewidth=0.5, label = "CI")
        ax[count].plot(x, iv_l, "g--", linewidth=0.5)

        #scatter points
        x = df[var_to_plot]
        y = df['spike_count']
        ax[count].plot(x, y, "o", label="data")

        ax[count].set_xlabel(var_to_plot, fontsize=12)
        ax[count].set_ylabel("Spike Count", fontsize=12)
        ax[count].set_title("Fitted Model with Confidence Bars", fontsize=14)
        ax[count].tick_params(axis='both', which='major', labelsize=12)
        ax[count].legend(loc="best", fontsize=8)
        ax[count].grid(True)

        count = count + 1

    plt.show()

def summary(model, final_features, X_test, y_test, sinusoidal = False):
    """
    Get summary results for the model

    Args:
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): _description_
        final_features (list): list of selected features with step_down
        X_test (numpy.ndarray): X test set 
        y_test (numpy.ndarray): y test set
    """
    get_result(model, X_test[:, final_features], y_test)
    _plot_residuals(model)
    print(get_equation(final_features, model, sinusoidal))
    
def plot_stats_neurons(adjr2_neuron1, adjr2_neuron2, adjr2_neuron3, adjr2_neuron4, adjr2_neuron5,
                       testloss_neuron1, testloss_neuron2, testloss_neuron3, testloss_neuron4, testloss_neuron5,
                       title, units):
    """
    Plots adjusted r2 and test rmse across models for each neuron in a separate plot

    Args:
        adjr2_neuron1 (list): _description_
        adjr2_neuron2 (list): _description_
        adjr2_neuron3 (list): _description_
        adjr2_neuron4 (list): _description_
        adjr2_neuron5 (list): _description_
        testloss_neuron1 (list): _description_
        testloss_neuron2 (list): _description_
        testloss_neuron3 (list): _description_
        testloss_neuron4 (list): _description_
        testloss_neuron5 (list): _description_
    """
    
    functions = ["linear", "quadratic", "sinusoidal", "combined"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9), sharex = True)
    ax1.plot(adjr2_neuron1, marker='o', color='r', label=units[0]) # neuron 1
    ax1.plot(adjr2_neuron2, marker='s', color='g', label=units[1]) # neuron 2
    ax1.plot(adjr2_neuron3, marker='^', color='b', label=units[2]) # neuron 3
    ax1.plot(adjr2_neuron4, marker='p', color='c', label=units[3]) # neuron 4
    ax1.plot(adjr2_neuron5, marker='v', color='m', label=units[4]) # neuron 5
    ax1.set_ylabel('Adjusted R Squared')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(functions)))
    ax1.set_xticklabels(functions)
    ax1.legend(loc = "best")
    ax1.grid(linestyle='--')
    plt.tight_layout()
    ax2.plot(testloss_neuron1, marker='o', color='r', label=units[0]) # neuron 1
    ax2.plot(testloss_neuron2, marker='s', color='g', label=units[1]) # neuron 2
    ax2.plot(testloss_neuron3, marker='^', color='b', label=units[2]) # neuron 3
    ax2.plot(testloss_neuron4, marker='p', color='c', label=units[3]) # neuron 4
    ax2.plot(testloss_neuron5, marker='v', color='m', label=units[4]) # neuron 5
    ax2.set_xlabel('Input Features')
    ax2.set_ylabel('Test RMSE')
    ax2.set_xticks(range(len(functions)))
    ax2.set_xticklabels(functions)
    ax2.legend(loc = "best")
    ax2.grid(linestyle='--')
    plt.suptitle(f"Neurons Selected by {title}")
    plt.tight_layout()
    plt.show()
    