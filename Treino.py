import pandas as pd

dataLocation = 'Dados\dados_vendas_brasil_2020_2025.csv'
targetCollumn = 'vendas_totais_no_mês'

df = pd.read_csv(dataLocation)

features = df.columns.tolist()
print(f'Features usadas: \n {features}')

target = df.get('vendas_totais_no_mês')
print(f'Coluna target: \n {target}')

