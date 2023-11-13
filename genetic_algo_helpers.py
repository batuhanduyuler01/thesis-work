from helpers import *
import numpy as np
import pandas as pd
import random
from enum import Enum

class FitnessType(Enum):
    F1_SCORE = 0
    LOG_MP_SUM = 1

class MatrixProfileCalculation(Enum):
    CLASSICAL = 0
    HACK = 1

class AlgorithmBase:
    def __init__(self):
        pass

    def calculate_cost(self, dataX:pd.DataFrame, dataY:pd.DataFrame):
        raise NotImplemented

    def min_max_normalize_dataframe(self, dataframe:pd.DataFrame):
        normalized_df : pd.DataFrame =(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
        return normalized_df.values


    def z_normalize_dataframe(self, dataframe:pd.DataFrame):
        means = dataframe.mean(axis=0)
        stds = dataframe.std(axis=0)
        z_normalized_data = (dataframe - means) / stds
        return z_normalized_data

    def euclidean_distance_dataframe(self, a:np.array, b:np.array):
        return np.sqrt(np.sum((a - b)**2))

class GeneticAlgorithm:
    def __init__(self,*, dataset:pd.DataFrame, number_of_feature:int, bag_size:int, algorithm:AlgorithmBase, fitness_type:FitnessType):
        self.df:pd.DataFrame = dataset.copy()
        self.num_feat:int = number_of_feature
        self.pop_bag_size:int = bag_size
        self.population_bag:list = []

        self.y = self.df[["Label"]]
        self.X = self.set_X(self.df.copy())
        self.feature_map = {i : feat_name for i, feat_name in enumerate(self.X.columns)}
        self.X.columns = list(range(0, len(self.X.columns)))

        self.algorithm : AlgorithmBase = algorithm
        self.fitness_type:FitnessType = fitness_type


    def set_X(self, dataframe:pd.DataFrame):
        drop_list = ["Label", "label", "Date_Second", " Flow Duration"]
        for d in drop_list:
            if d in dataframe.columns:
                dataframe.drop(d, axis=1, inplace=True)

        return dataframe
    
    def initialize_population(self):
        self.population_bag.clear()
        for _ in range(self.pop_bag_size):
            genes = [random.randrange(0,2) for _ in range(self.num_feat)]
            gene_indexes = [idx for idx, f in enumerate(genes) if f == 1]
            if (len(gene_indexes) == 0):
                gene_indexes.append(random.randint(1, self.num_feat))
            
            self.population_bag.append(self.X.iloc[:, gene_indexes])

        return self.population_bag
    
    def create_population(self, best_population_indexes:list) -> pd.DataFrame:
        self.population_bag.clear()
        for elem in best_population_indexes:
            self.population_bag.append(self.X.iloc[:, elem])
            
        return self.population_bag
    
    def fitness_function(self, individual:pd.DataFrame):
        cost, f1_score = self.algorithm.calculate_cost(individual, self.y)
        return (cost, f1_score)
    
    def eval_fit_population(self, pop_bag:list):
        #Minimize The Cost!
        
        result, fit_vals_list, f1_score_list, solutions = {}, [], [], []
        for individual in pop_bag:
            assert(type(individual) != type(pd.DataFrame))
            cost, f1_sc = self.fitness_function(individual.copy())
            fit_vals_list.append(cost)
            f1_score_list.append(f1_sc)
            solutions.append(individual.columns.to_list())
        
        result["fit_vals"] = fit_vals_list
        result["f1_scores"] = f1_score_list

        if self.fitness_type == FitnessType.LOG_MP_SUM:
            min_wgh = [abs(np.min(list(result['fit_vals'])) - i) for i in list(result['fit_vals'])]
        elif (self.fitness_type == FitnessType.F1_SCORE):
            min_wgh = [abs(np.min(list(result['f1_scores'])) - i) for i in list(result['f1_scores'])]

        from scipy.special import logsumexp
        #TODO: find a way instead of logsumexp
        result["fit_wgh"]  = [i/logsumexp(min_wgh) for i in min_wgh]
        result["solution"] = np.array(solutions, dtype=list).tolist()
        
        self.eval_result = result.copy()
        return result

    def find_best(self, eval_result:dict)->dict:
        # Best individual so far
        max_idx = [(np.max(val), i) for (i,val) in enumerate(eval_result['fit_vals'])]
        mP, mI = sorted(max_idx, key=lambda x:x[1], reverse=True)[0]
        best_fit_index = eval_result["fit_vals"][mI].index(mP)
        best_solution  = eval_result["solution"][mI]
        f1_sc = eval_result["f1_scores"][mI]
 
        return {'best_fit': mP, 'index' : mI,
                 'solution': best_solution, 'f1-score' : f1_sc}    


    def pick_one(self, pop_bag):
        
        if self.eval_result is None:
            eval_result = self.eval_fit_population(pop_bag)
        else:
            eval_result = self.eval_result

        notPicked=True
        cnt = 0
        pickedSol = list()
        while (notPicked == True):
            rnIndex = random.randint(0, len(pop_bag)-1)
            rnPick  = eval_result["fit_wgh"][rnIndex]
            r = random.random()
            if  r <= rnPick:
                pickedSol = eval_result["solution"][rnIndex]
                notPicked = False
            if (cnt > 250):
                pickedSol = eval_result["solution"][rnIndex]
                notPicked = False
            cnt += 1

        return pickedSol
    
    def crossover(self, solA, solB):
        
        n     = len(solA)
        child: list = []

        num_els = random.randint(0, self.num_feat)
        str_pnt = random.randint(0, max(0,n-3))
        end_pnt = n if int(str_pnt+num_els) > n else int(str_pnt+num_els)

        blockA = list(solA[str_pnt:end_pnt])
        child = blockA.copy()

        for elem in solB:
            if len(child) >= num_els:
                break
            if elem not in blockA:
                child.append(elem)  

        if (len(child) < 1):
            return solA

        return child

    def mutation(self,sol):
        if (len(sol) > 2):
            rd_idx = random.randint(0, len(sol) - 1)
            del sol[rd_idx]
        return sol
        

class MatrixProfile(AlgorithmBase):
    def __init__(self, window_size:int, matrix_profile_calculation:MatrixProfileCalculation = MatrixProfileCalculation.CLASSICAL):
        self.window_size:int = window_size
        self.mp_calculation = matrix_profile_calculation
        
        #For Debugging
        self.mp = None
        self.z_data = None 

    def compute_matrix_profile_majority(self, dataframe:pd.DataFrame, window_size:int):
        import matrixprofile as mp
        mp_list = np.array([0.0 for _ in range(len(dataframe) - window_size + 1)])
        
        for ft in dataframe.columns:
            inputSignal = dataframe[ft].to_list()
            matrix_profile = mp.compute(inputSignal, windows=window_size, threshold=0.95, n_jobs=4)
            mp_list += (np.array(matrix_profile['mp'])**2)
        
        return np.sqrt(mp_list), []

    def compute_matrix_profile(self, dataframe:pd.DataFrame, window_size:int):
        num_rows = dataframe.shape[0]
        # normalized_data = self.z_normalize_dataframe(dataframe.values)
        normalized_data = self.min_max_normalize_dataframe(dataframe=dataframe)
        # normalized_data = dataframe.values
            
        # Initialize an empty matrix profile with large values
        matrix_profile = np.full(num_rows - window_size + 1, np.inf)

        # Initialize an empty matrix profile index
        matrix_profile_index = np.zeros(num_rows - window_size + 1, dtype=int)

        for i in range(num_rows - window_size + 1):
            subsequence = normalized_data[i:i+window_size]

            for j in range(num_rows - window_size + 1):
                if i != j:
                    candidate_subsequence = normalized_data[j:j+window_size]
                    distance = self.euclidean_distance_dataframe(subsequence, candidate_subsequence)

                    if distance < matrix_profile[i]:
                        matrix_profile[i] = distance
                        matrix_profile_index[i] = j

        self.mp = matrix_profile.copy()
        self.z_data = normalized_data.copy()
        return matrix_profile, matrix_profile_index
    
    def calculate_matrix_profile_window_labels(self, dataY)->pd.Series:
        mp_window_num = len(dataY) - self.window_size + 1 
        label_list = []

        for w in range(0, mp_window_num):

            if (sum(dataY.iloc[w:w+self.window_size, -1]) > 0): #indicates label
                label_list.append(1)
            else:
                label_list.append(0)

        return pd.Series(label_list)
    
    def matrix_profile_preparation(self, dataX:pd.DataFrame, dataY:pd.DataFrame)->pd.DataFrame:
        
        if (self.mp_calculation == MatrixProfileCalculation.CLASSICAL):
            matrix_profile, w_indexes = self.compute_matrix_profile(dataX, self.window_size)
        elif (self.mp_calculation == MatrixProfileCalculation.HACK):
            matrix_profile, w_indexes = self.compute_matrix_profile_majority(dataX, self.window_size)

        windowNumber:int = len(matrix_profile)
        resultData = pd.DataFrame({"W_Label":[0 for _ in range(windowNumber)], "MP_scores": [0 for _ in range(windowNumber)], "Preds": [0 for _ in range(windowNumber)]})
        resultData['MP_scores'] = pd.Series(matrix_profile)
        resultData['W_Label']  = self.calculate_matrix_profile_window_labels(dataY)
        return resultData

    def calculate_cost(self, dataX:pd.DataFrame, dataY:pd.DataFrame):
        raise NotImplemented


class MatrixProfileThreshold(MatrixProfile):
    def __init__(self, window_size:int, threshold:float,  matrix_profile_calculation:MatrixProfileCalculation):
        super().__init__(window_size, matrix_profile_calculation) 
        self.threshold = threshold

    def calculate_cost(self, dataX:pd.DataFrame, dataY:pd.DataFrame):

        resultData:pd.DataFrame = self.matrix_profile_preparation(dataX.copy(), dataY.copy())
        anomalies = resultData[resultData["MP_scores"] > self.threshold].index.to_list()
        resultData.iloc[:, -1] = 0
        resultData.iloc[anomalies, -1] = 1
        self.rData = resultData.copy()
        from sklearn.metrics import classification_report
        creport =  classification_report(resultData["W_Label"], resultData["Preds"], output_dict=True)
        return (resultData["MP_scores"].to_list(), creport["macro avg"]['f1-score'])



class MatrixProfileIQR(MatrixProfile):
    def __init__(self, window_size:int, iqr_coefficient:int, matrix_profile_calculation:MatrixProfileCalculation):
        super().__init__(window_size, matrix_profile_calculation)
        self.iqr_coefficient = iqr_coefficient

    def calculate_cost(self, dataX: pd.DataFrame, dataY: pd.DataFrame):
        resultData:pd.DataFrame = self.matrix_profile_preparation(dataX.copy(), dataY.copy())
        Q1 = resultData["MP_scores"].quantile(0.25)
        Q3 = resultData["MP_scores"].quantile(0.75)
        IQR = Q3 - Q1
        self.threshold = Q3 + (self.iqr_coefficient * IQR)
        anomalies = resultData[resultData["MP_scores"] > self.threshold].index.to_list()
        resultData.iloc[:, -1] = 0
        resultData.iloc[anomalies, -1] = 1
        self.rData = resultData.copy()
        from sklearn.metrics import classification_report
        creport =  classification_report(resultData["W_Label"], resultData["Preds"], output_dict=True)
        return (resultData["MP_scores"].to_list(), creport["macro avg"]['f1-score'])
        

class MatrixProfilePercentile(MatrixProfile):
    def __init__(self, window_size:int, percentile:int, matrix_profile_calculation:MatrixProfileCalculation):
        super().__init__(window_size, matrix_profile_calculation)
        self.percentile = percentile
    
    def calculate_cost(self, dataX: pd.DataFrame, dataY: pd.DataFrame):
        resultData:pd.DataFrame = self.matrix_profile_preparation(dataX.copy(), dataY.copy())
        anomalies = resultData[resultData["MP_scores"] > np.percentile(resultData["MP_scores"], self.percentile)].index.to_list()
        resultData.iloc[:, -1] = 0
        resultData.iloc[anomalies, -1] = 1
        self.rData = resultData.copy()
        from sklearn.metrics import classification_report
        creport =  classification_report(resultData["W_Label"], resultData["Preds"], output_dict=True)
        return (resultData["MP_scores"].to_list(), creport["macro avg"]['f1-score'])
        

class MatrixProfileIsolation(MatrixProfile):
    def __init__(self, window_size:int, n_est:int, cont = 'auto',  matrix_profile_calculation:MatrixProfileCalculation = MatrixProfileCalculation.HACK):
        super().__init__(window_size, matrix_profile_calculation)         
        self.cont = cont
        self.n_est = n_est

    def calculate_cost(self, dataX: pd.DataFrame, dataY: pd.DataFrame):
        resultData:pd.DataFrame = self.matrix_profile_preparation(dataX.copy(), dataY.copy())

        from sklearn.ensemble import IsolationForest

        model=IsolationForest(n_estimators = self.n_est, contamination = self.cont)
        model.fit(np.array(resultData["MP_scores"]).reshape(-1, 1))
        data = resultData[["MP_scores"]].copy()
        data['scores']=model.decision_function(np.array(resultData["MP_scores"]).reshape(-1,1))
        data['anomaly']=model.predict(np.array(resultData["MP_scores"]).reshape(-1,1))
        resultData["Preds"] = data["anomaly"].apply(lambda x: 1 if x == -1 else 0)  

        self.rData = resultData.copy()
        from sklearn.metrics import classification_report
        creport =  classification_report(resultData["W_Label"], resultData["Preds"], output_dict=True)
        return (resultData["MP_scores"].to_list(), creport["macro avg"]['f1-score'])



def eliminate_nan_cols(dataframe:pd.DataFrame)->pd.DataFrame:
    for col in dataframe.columns:
        if (sum(dataframe[col].isna()) > 0):
            dataframe.drop(col, axis=1, inplace=True)

    return dataframe


def fetch_data():
    data_paths = {  'd_ntp'     : '../jupyter-notebooks/ntp_ddos_14_minutes.csv',
                    'd_udp'     : '../jupyter-notebooks/udp_ddos_2_minutes.csv',
                    'd_syn'     : '../jupyter-notebooks/syn_ddos_3_minutes.csv',
                    'd_ldap'    : '../jupyter-notebooks/ldap_ddos_9_minutes.csv',
                    'd_udp_lag' : '../jupyter-notebooks/udp_lag_ddos_7_minutes.csv',
                
                    'b_ntp'     : '../jupyter-notebooks/ntp_benign_30_minutes.csv',
                    'b_syn'     : '../jupyter-notebooks/syn_benign_1_minutes.csv',
                    'b_ldap'    : '../jupyter-notebooks/ldap_benign_5_minutes.csv',
                    'b_udp_lag' : '../jupyter-notebooks/udp_lag_benign_7_minutes.csv'}

    dataset_dict = {}

    for data_name, path in data_paths.items():
        data = upload_dataset_with_time(path)
        dataset_dict[data_name] = data


    syn_df = pd.concat([dataset_dict["b_ldap"], dataset_dict["b_ldap"] , dataset_dict["d_syn"],
                        dataset_dict["d_syn"], dataset_dict["b_udp_lag"],
                        dataset_dict["d_syn"], dataset_dict["b_ntp"].iloc[:5*60, :]], axis=0).reset_index(drop=True)

    ntp_df = pd.concat([dataset_dict["b_ldap"], dataset_dict["b_ldap"] , dataset_dict["d_ntp"][:5*60],
                        dataset_dict["b_udp_lag"],
                        dataset_dict["d_syn"], dataset_dict["b_ntp"].iloc[:5*60, :]], axis=0).reset_index(drop=True)


    mixed_attacks = pd.concat([dataset_dict["b_ldap"], dataset_dict["b_ldap"],
                            dataset_dict["d_ntp"][:1*60], #DDOS
                            dataset_dict["b_udp_lag"],
                            dataset_dict["d_syn"][:1*60], #DDOS
                            dataset_dict["b_ntp"][:3*60],
                            dataset_dict["d_udp_lag"][:1*60], #DDOS
                            dataset_dict["b_ntp"][:5*60],
                            dataset_dict["d_syn"][:1*60], #DDOS
                            dataset_dict["b_syn"][:1*60],
                            dataset_dict["d_udp"][:1*60], #DDOS
                            dataset_dict["b_udp_lag"][:4*60],
                            dataset_dict["d_ntp"][:1*60], #DDOS,
                            dataset_dict["b_ldap"][:3*60],
                            dataset_dict["d_ldap"][:1*60], #DDOS
                            dataset_dict["b_ldap"], dataset_dict["b_ntp"][:5*60],
                            dataset_dict["d_ntp"][:1*60], #DDOS
                            dataset_dict["b_syn"],
                            dataset_dict["d_syn"][:1*60], #DDOS
                            dataset_dict["b_ldap"], dataset_dict["b_ldap"][:2*60]], axis = 0).reset_index(drop=True)
    
    return {'syn_df': syn_df.copy(), 'ntp_df' : ntp_df.copy(), 'mixed_df': mixed_attacks.copy()}