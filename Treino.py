from model import createModel
from dados import split_data
from pred import predict_next_six_months

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def trainFunction(csv_path, targetCollumn, test_size, random_state, device, learnRate, epochs, patience, patience_counter = 0):
    
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = split_data(csv_path, targetCollumn,test_size, random_state)
    print('oi3')
    input_dim = X_train.shape[1]
    model = createModel(input_dim)
    model.to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    print(f'Modelo usando: {next(model.parameters()).device}')

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                learnRate)

    best_test_loss = float('inf')
                            
    
    for epoch in range(epochs):

        model.train()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.squeeze(1), y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval() 

        with torch.inference_mode(): 
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred.squeeze(1), y_test)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0  
            torch.save(model.state_dict(), 'Model/model.pth')
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping on epoch {epoch}, no improvement for {patience} epochs.")
                break

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Training Loss: {loss.item()} | Test Loss: {test_loss.item()}')

    predict_next_six_months(csv_path, model, scaler_X, scaler_y, device)