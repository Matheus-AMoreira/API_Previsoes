import torch
import torch.nn as nn

def createModel(input_dim):

    # Define the Linear Regression Model
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1) # One input feature, one output

        def forward(self, x):
            return self.linear(x)

    return LinearRegressionModel(input_dim)