"""
Makes predictions using an existing model.

:author: Max Milazzo
:email: mam9563@rit.edu
"""


import sys
import string
import pickle


def predict(model_file: str, prediction_atts: list) -> str:
    """
    Makes a prediction given a model and attributes.
    """
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
        # load model
        
    return model.run(prediction_atts)


def main(args: list) -> None:
    """
    Program entry point.
    """
    
    predict(args[0], args[1])
    

if __name__ == "__main__":
    num_args = 3
    args = sys.argv
    # gets arguments and defines expected argument length
    
    if len(args) == num_args:
        main(args[1:])
        # calls main with command line arguments
        
    else:
        print("usage: py predict.py <hypothesis> <file>")