import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do banco de dados
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Modelo da tabela de previsões
class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    categoria = Column(String(100), nullable=False)
    ano = Column(Integer, nullable=False)
    mes = Column(Integer, nullable=False)
    quantidade = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

def create_prediction_table():
    """Cria a tabela de previsões se não existir."""
    Base.metadata.create_all(engine)

def split_data(df, target_column, test_size=0.2, random_state=42, lag_features=3):
    """Prepara os dados com lags e divide em treino e teste."""
    for lag in range(1, lag_features + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    
    df = df.dropna().reset_index(drop=True)
    
    X = df[[f'lag_{i}' for i in range(1, lag_features + 1)]].values
    y = df[target_column].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler_X,
        scaler_y
    )

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(df, target_column, test_size=0.2, random_state=42, device='cpu', learn_rate=0.01, epochs=1000, patience=20):
    """Treina o modelo de regressão linear."""
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = split_data(df, target_column, test_size, random_state)
    
    input_dim = X_train.shape[1]
    model = LinearRegressionModel(input_dim).to(device)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    best_test_loss = float('inf')
    patience_counter = 0
    
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
            torch.save(model.state_dict(), 'model.pth')
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping on epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Training Loss: {loss.item()} | Test Loss: {test_loss.item()}')
    
    return model, scaler_X, scaler_y

def predict_next_six_months(df, model, scaler_X, scaler_y, device, categoria, lag_features=3):
    """Faz previsões para os próximos seis meses."""
    model.eval()
    last_data = df.tail(lag_features)[['quantidade']].values.flatten()
    
    predictions = []
    current_date = datetime.now()
    
    for i in range(6):
        input_data = last_data[-lag_features:]
        input_data = scaler_X.transform(input_data.reshape(1, -1))
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        
        with torch.inference_mode():
            pred = model(input_tensor)
        pred = scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()[0]
        pred = max(0, int(round(pred)))
        
        forecast_date = current_date + relativedelta(months=i + 1)
        ano = forecast_date.year
        mes = forecast_date.month
        
        predictions.append({
            'categoria': categoria,
            'ano': ano,
            'mes': mes,
            'quantidade': pred
        })
        
        last_data = np.append(last_data, pred)[-lag_features:]
    
    return predictions

def update_predictions(df, categoria, device='cpu'):
    """Atualiza previsões no banco para a categoria especificada."""
    if len(df) < 4:  # 3 lags + 1 target
        raise ValueError("Dados insuficientes para treinar o modelo (mínimo 4 meses).")
    
    model, scaler_X, scaler_y = train_model(
        df=df,
        target_column='quantidade',
        test_size=0.2,
        random_state=42,
        device=device
    )
    
    predictions = predict_next_six_months(
        df=df,
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        device=device,
        categoria=categoria
    )
    
    session = Session()
    try:
        # Deletar previsões existentes para a categoria, ano e mês
        for pred in predictions:
            session.query(Prediction).filter(
                Prediction.categoria == pred['categoria'],
                Prediction.ano == pred['ano'],
                Prediction.mes == pred['mes']
            ).delete()
        
        # Inserir novas previsões
        for pred in predictions:
            prediction = Prediction(
                categoria=pred['categoria'],
                ano=pred['ano'],
                mes=pred['mes'],
                quantidade=pred['quantidade']
            )
            session.add(prediction)
        session.commit()
        return predictions
    except Exception as e:
        session.rollback()
        raise Exception(f"Erro ao atualizar previsões: {str(e)}")
    finally:
        session.close()