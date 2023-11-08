import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matrixprofile as mp
import random
import time
import os
from matplotlib.patches import Rectangle
import lmoments3

def upload_dataset_with_time(path:str):
    startingT = time.perf_counter()
    if 'pkl' in path:
        veriseti = pd.read_pickle(path)
    else:
        veriseti = pd.read_csv(path, low_memory=True)
    endingT = time.perf_counter()
    print(f"Dataset is loaded in {endingT - startingT} seconds")
    return veriseti

def plot_ddos(df: pd.DataFrame, attack_list):
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='lightgreen', lw=3),
                    Line2D([0], [0], color='red', lw=3),
                    Line2D([0], [0], color='blue', lw=3),
                    Line2D([0], [0], color='black', lw=3),
                    Line2D([0], [0], color='brown', lw=3)]
    
    attack_color_dict = {'syn' : 'lightgreen', 'ntp' : 'red', 'udp' : 'blue', 'udp_lag' : 'black', 'ldap' : 'brown'}
    
    total_duration = len(df) / 60
    print(f"Total Duration Of Traffic is: {total_duration} minutes")
    xAxis = list(range(len(df)))
    yAxis = df["Label"].to_list()
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot()
    ax.plot(xAxis, yAxis)
    
    legend_custom_lines = []
    legend_custom_names = []
    for attack in attack_list:
        face_color = attack_color_dict[attack[0]]
        attack_duration = attack[1]
        duration_before_attack = attack[2]
        
        attack_index = list(attack_color_dict.keys()).index(attack[0])
        if attack[0] not in legend_custom_names:
            legend_custom_lines.append(custom_lines[attack_index])
            legend_custom_names.append(attack[0])
        
        rect = Rectangle((duration_before_attack * 60, 0), attack_duration * 60, 1, facecolor=face_color)
        ax.add_patch(rect)

    
    ax.legend(legend_custom_lines, legend_custom_names,  prop={'size': 20})
    
    plt.ylabel('Label')
    plt.xlabel('Seconds')
    plt.title('Network Traffic')
    plt.show()

def extract_lmoms_of_data(data :pd.DataFrame,feat_idx : int, windowSize : int, nmom_num:int=4)->pd.DataFrame:
    from collections import Counter
    lmom_dict = {   'L_1' : [],
                    'L_2' : [],
                    'L_3' : [],
                    'L_4' : [],
                    'W_Label' : []}
    
    lmoment_window_number = len(data) - windowSize + 1 

    for w in range(0, lmoment_window_number):
        temp_w = data.iloc[w: w+windowSize, feat_idx].to_list()
        temp_l_mom = lmoments3.lmom_ratios(temp_w, nmom=nmom_num)
        for mom in range(1,nmom_num+1):
            lmom_dict[f"L_{mom}"].append(temp_l_mom[mom - 1])
    
        if (sum(data.iloc[w:w+windowSize, -1]) > 0): #indicates label
            lmom_dict["W_Label"].append(1)
        else:
            lmom_dict["W_Label"].append(0)

    labelCounter = Counter(lmom_dict["W_Label"])
    print(f"total benign window: {labelCounter[0]}")
    print(f"total ddos window: {labelCounter[1]}")

    return pd.DataFrame(lmom_dict)



def plot_lmom_scatter_of_data(lmom_df, lmom_num):
    num_dict = {1: 'L - Location', 2: 'L - Scale', 3: 'L - Skewness', 4: 'L - Kurtosis'}
    if lmom_num > 5:
        raise ValueError('LMOM Number can not be greater than 5!')
    
    color_list = lmom_df["W_Label"].apply(lambda x:'red' if x == 1 else 'blue').to_list()
    plt.figure(figsize=(20,10))
    plt.scatter(list(range(0,len(lmom_df))), lmom_df[f"L_{lmom_num}"], color=color_list)
    plt.title(num_dict[lmom_num])
    plt.xlabel('Window Number')
    plt.ylabel(f'L_{lmom_num}')
    plt.show()

def calculate_matrix_profile_window_labels(preprocessed_data, windowSize, labelIdx):
    mp_window_num = len(preprocessed_data) - windowSize + 1 
    label_dict = {'Label': []}

    for w in range(0, mp_window_num):

        if (sum(preprocessed_data.iloc[w:w+windowSize, labelIdx]) > 0): #indicates label
            label_dict["Label"].append(1)
        else:
            label_dict["Label"].append(0)

    return pd.DataFrame(label_dict)


class df_feature_controller:
    def __init__(self, df:pd.DataFrame):
        self.ft_names = df.columns.to_list()
        self.ft_list = list(zip(list(range(0, len(df))), self.ft_names))

    def get_feature_index(self, feature_name):
        if feature_name not in self.ft_names:
            raise ValueError("this feat not in df cols")
        
        for i in range(0, len(self.ft_list)):
            if self.ft_list[i][1] == feature_name:
                break
        return i

    def get_feature_name(self, feat_idx):
        print(f" {self.ft_list[feat_idx][1]} is the feature of idx {feat_idx}")
        return self.ft_list[feat_idx][1]
    
def isolation_forest_predictions(data, n_est=200, cont=0.1):
    from sklearn.ensemble import IsolationForest

    model=IsolationForest(n_estimators = n_est, contamination = cont)
    model.fit(data)
        
    data['scores']=model.decision_function(data.iloc[:,:])
    data['anomaly']=model.predict(data.iloc[:, :-1])
    return data


def calculate_lmom_df(data, selected_feature, windowSize):
    feat_ctrl = df_feature_controller(data)
    lmom_df = extract_lmoms_of_data(data=data.copy(), feat_idx=feat_ctrl.get_feature_index(selected_feature), windowSize=windowSize)
    lmom_df = lmom_df.replace(np.nan, 0.001)
    return lmom_df

def lmom_predictions(df, skew_th = 15, kurt_th = 15, verbose=False):
    lmom_classifier_results = df.copy()
    lmom_classifier_results["Preds"] = 0

    skew_threshold = np.percentile(lmom_classifier_results["L_3"], skew_th)
    kurt_threshold = np.percentile(lmom_classifier_results["L_4"], kurt_th)


    lmom_anomalies_skew = lmom_classifier_results[lmom_classifier_results["L_3"] <= skew_threshold].index.to_list()
    lmom_anomalies_kurtosis = lmom_classifier_results[lmom_classifier_results["L_4"] <=  kurt_threshold].index.to_list()

    lmom_anomalies = np.union1d(lmom_anomalies_skew, lmom_anomalies_kurtosis)

    lmom_classifier_results.iloc[lmom_anomalies, -1] = 1

    from sklearn.metrics import classification_report
    if (verbose):
        print(f"kurt_thresh = {kurt_threshold}\nskew_thresh = {skew_threshold}")
        print(f"Number of anomalies detected: {len(lmom_anomalies)}")
        print(classification_report(lmom_classifier_results["W_Label"], lmom_classifier_results["Preds"]))

    return classification_report(lmom_classifier_results["W_Label"], lmom_classifier_results["Preds"], output_dict=True)


def calculate_matrix_profile(data, selected_feature, windowSize):
    feat_ctrl = df_feature_controller(data)
    mp_data = mp.compute(data.iloc[:, feat_ctrl.get_feature_index(selected_feature)].to_list(), windowSize)
    return mp_data

def matrix_profile_threshold_predictions(data:pd.DataFrame, mp_threshold=0, mp_threshold_percentile=0 ,iqr_coefficient=0, verbose=False):
    mp_classifier_results = data.copy()
    mp_classifier_results["Preds"] = 0

    if (mp_threshold > 0) and (mp_threshold_percentile > 0):
        raise ValueError("Both Thresholds can not used at same time")
    
    anomalies = []
    if (mp_threshold > 0):
        if verbose:    
            print(f"threshold is mp base : {mp_threshold}")
        anomalies = mp_classifier_results[mp_classifier_results["mp_scores"] > mp_threshold].index.to_list()

    if (mp_threshold_percentile > 0 ):
        if verbose:
            print(f"threshold is percentile based: {mp_threshold_percentile}")
        anomalies = mp_classifier_results[mp_classifier_results["mp_scores"] > np.percentile(mp_classifier_results["mp_scores"], mp_threshold_percentile)].index.to_list()
    
    if (iqr_coefficient > 0):
        Q1 = mp_classifier_results["mp_scores"].quantile(0.25)
        Q3 = mp_classifier_results["mp_scores"].quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + (iqr_coefficient) * IQR
        anomalies = mp_classifier_results[mp_classifier_results["mp_scores"] > threshold].index.to_list()



    mp_classifier_results.iloc[anomalies, -1] = 1
    if verbose:
        print(anomalies)
        print(mp_classifier_results)


    from sklearn.metrics import classification_report
    if (verbose):
        print(classification_report(mp_classifier_results["W_Label"], mp_classifier_results["Preds"]))
        print(f"Number of anomalies detected: {len(anomalies)}")

    return classification_report(mp_classifier_results["W_Label"], mp_classifier_results["Preds"], output_dict=True)


def isolation_forest_predictions(data, n_est=200, cont=0.1):
    from sklearn.ensemble import IsolationForest

    model=IsolationForest(n_estimators = n_est, contamination = cont)
    model.fit(data.iloc[:,:])
        
    data['scores']=model.decision_function(data.iloc[:,:])
    data['anomaly']=model.predict(data.iloc[:, :-1])
    return data