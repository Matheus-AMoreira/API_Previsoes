import sys
print(sys.executable)

from dataGenerate import GenerateData
from Treino import trainFunction

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

print('oi')

csv_path = 'Dados/toy_sales_2010-2024.csv'
targetCollumn = 'Units_Sold'

test_size=0.2
random_state=42

device = "cuda" if torch.cuda.is_available() else "cpu"

learnRate = 0.001

epochs=10000
patience=10

trainFunction(csv_path, targetCollumn, test_size, random_state, device, learnRate, epochs, patience)