# analyzing the training data set for the comment classification challenge

import pandas as pd
import numpy as np
import os
import re

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))), 'Data')

print(DATA_PATH)


# load the data
data = pd.read_csv('train.csv')
