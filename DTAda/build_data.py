"""
Takes a formatted dataset and creates a data list.

:author: Max Milazzo
:email: mam9563@rit.edu
"""


import string
from random import random


def set_default_weights(train_set: list) -> list:
    """
    Sets default weights for training set such that they sum to 1.
    """
    
    weight = 1 / len(train_set)
    
    for x in range(len(train_set)):
        train_set[x].append(weight)
        train_set[x] = train_set[x]
        
    return train_set


def train_test_split(data: list, test_set_faction: float) -> tuple:
    """
    Splits training and testing data.
    """
    
    train_set = []
    test_set = []
    
    for elem in data:
        if random() < test_set_faction:
            test_set.append(elem)
        else:
            train_set.append(elem)
    
    return train_set, test_set
    

def build_data(file: str) -> list:
    """
    Builds data list from file.
    """
    
    data = []
    
    cat1 = None
    cat2 = None
    
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line_dat = line.split()
            category = line_dat[0]
            line_dat = line_dat[1:]
            # fetches category and attribute data from file
            
            if cat1 is None:
                cat1 = category
            elif cat2 is None and category != cat1:
                cat2 = category
            
            atts = []
            
            for elem in line_dat:
                if elem.lower() == "true" or elem.lower() == "1":
                    atts.append(True)   
                else:
                    atts.append(False)
            

            data.append([atts, category])
            
    return data, cat1, cat2