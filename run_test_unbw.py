from helpers import *
from genetic_algo_helpers import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from time import perf_counter

import argparse

if __name__ == "__main__":
    algorithm = None
    choices = ['mp-threshold', 'mp-iqr', 'mp-percentile', 'mp-isolation']
    parser = argparse.ArgumentParser(description='Run MP with Genetic Algorithm')
    parser.add_argument('--mp', type=str, help='Run MP with Genetic Algorithm', choices=choices, required=True)

    args = parser.parse_args()
    if args.mp.lower() == 'mp-iqr':
        algorithm = MatrixProfileIQR(20, 2.0, MatrixProfileCalculation.HACK)
    elif args.mp.lower() == 'mp-threshold':
        algorithm = MatrixProfileThreshold(20, 2.0, MatrixProfileCalculation.HACK)
    elif args.mp.lower() == 'mp-percentile':
        algorithm = MatrixProfilePercentile(20, 70, MatrixProfileCalculation.HACK)
    elif args.mp.lower() == 'mp-isolation':
        algorithm = MatrixProfileIsolation(20, 50, 'auto', MatrixProfileCalculation.HACK)
    else:
        print('Unknown Argument')
        exit()

    if algorithm is None:
        print('Bad Algo Request')
        exit()

        
    unbsw_dataset = pd.read_csv("../../Boun_Project/verisetleri/unsw_nb15/UNSW_NB15_training-set.csv")
    new_data = pd.concat([unbsw_dataset[unbsw_dataset.attack_cat == 'Normal'], 
                        unbsw_dataset[unbsw_dataset.attack_cat == 'Exploits']], axis = 0).reset_index(drop=True)

    new_data = new_data.sort_values(by='id').reset_index(drop=True)
    df = pd.concat([new_data.iloc[15000:, :], new_data.iloc[:15000, :]], axis=0).reset_index(drop=True)

    from genetic_algo_helpers import eliminate_nan_cols
    data = eliminate_nan_cols(df)
    df = data.rename({'label':"Label"}, axis=1)
    df = df.select_dtypes(['number']).iloc[:, 3:]
    print(df.head(10))
    print(df.columns)

    start_time = perf_counter()
    GA = GeneticAlgorithm(dataset=df.copy(), number_of_feature=20, bag_size=10,
                        algorithm=algorithm, fitness_type=FitnessType.F1_SCORE)

    pop_bag = GA.initialize_population()
    generation_number = 100

    import random
    f1_score_list = []
    for generation in range(generation_number):
        if (generation % 2 == 0):
            print(f"Generation {generation} is started!")
            
        res = GA.eval_fit_population(pop_bag)
        best_fit, _, best_solution, best_f1_score = GA.find_best(res).values()
        f1_score_list.append(best_f1_score)
            
        if (generation == 0):
            best_fit_global      = best_fit
            best_solution_global = best_solution
            best_f1_global       = best_f1_score
        else:
            if (best_f1_score >= best_f1_global):
                best_fit_global      = best_fit
                best_f1_global       = best_f1_score
                best_solution_global = best_solution
                print(f"best f1: {best_f1_global}")

        new_pop_bag = []
        for i in range(len(GA.population_bag)):
            # Pick 2 parents from the bag
            pA = GA.pick_one(pop_bag)
            pB = GA.pick_one(pop_bag)
            new_element = pA
            # Crossover the parents
            if random.random() <= 0.70:
                new_element = GA.crossover(pA, pB)
            #Mutate the child
            if random.random() <= 0.5:
                new_element = GA.mutation(new_element) 
                
            # Append the child to the bag
            new_pop_bag.append(new_element)
                # Set the new bag as the population bag
        pop_bag = GA.create_population(new_pop_bag)
            

    print("\n\n**** Generations Over ****\n")
    print(f"Best Fitness: {best_fit_global}")
    print(f"Best Solution: {best_solution_global}")
    print(f"F1-Score: {max(f1_score_list)}")

    end_time = perf_counter()
    print(f"simulation longs {end_time - start_time} seconds")
