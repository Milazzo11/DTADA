"""
Train new model.

:author: Max Milazzo
:email: mam9563@rit.edu
"""


import DTAda.build_data as build_data
from DTAda.build_DT import build_DT
from DTAda.build_ADA import build_ADA
import sys
import pickle


def build_model(mode: str, data: list, depth: int, cat1: str, cat2: str):
    """
    Builds a model.
    """
    
    if mode == "dt":
        model = build_DT(data, depth)
    else:
        model = build_ADA(data, cat1, cat2, depth)
        
    return model


def model_eval(data_file: str, mode: str, depth: int, num_models: int,
        test_set_faction: float) -> float:
    """
    Generates several models and finds the average accuracy as an evaluation.
    """

    accuracy_total = 0
    
    for x in range(num_models):
        print(f"MODEL {x + 1} of {num_models}:")
        print("Fetching data...")
        
        data, cat1, cat2 = build_data.build_data(data_file)
        train_set, test_set = build_data.train_test_split(data, test_set_faction)
        train_set = build_data.set_default_weights(train_set)
        # builds formatted training/testing data lists
        
        print("Training model...")
        model = build_model(mode, train_set, depth, cat1, cat2)
            
        print("Evaluating model...")
        accuracy = model.eval(test_set)
        
        print(f"Accuracy: {round(accuracy * 100, 2)}%\n")
        accuracy_total += accuracy

    print(f"Mean accuracy of trained models:")
    print(f"{round((accuracy_total / num_models) * 100, 2)}%\n")
    
    return accuracy_total / num_models
    

def save_model(data_file: str, save_file: str, mode: str, depth: int) -> None:
    """
    Builds and saves the final model using all available data.
    """
    
    data, cat1, cat2 = build_data.build_data(data_file)
    data = build_data.set_default_weights(data)
    # builds full formatted dataset
    
    print("Generating final model...")
    model = build_model(mode, data, depth, cat1, cat2)
    model.minimize()
    
    if save_file is not None:
        print("Saving model...")
        
        with open(save_file, "wb") as f:
            pickle.dump(model, f)
            
        print(f'Model "{save_file}" saved.')
    
    return model


def train(data_file: str, mode: str, depth: int = 5, num_models: int = 10,
        save_file: str = None, test_set_faction: float = 0.1) -> None:
    """
    Train new model.
    """
    
    accuracy = model_eval(data_file, mode, depth, num_models, test_set_faction)
    return save_model(data_file, save_file, mode, depth), accuracy


def main(args: list) -> None:
    """
    Program entry point.
    """
    
    train(args[0], args[1], 5, 10, args[2])

 
if __name__ == "__main__":
    num_args = 4
    args = sys.argv
    # gets arguments and defines expected argument length
    
    if len(args) == num_args:
        main(args[1:])
        # calls main with command line arguments
        
    else:
        print("usage: py train.py <examples> <learning-type> <hypothesisOut>")