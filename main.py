from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import func, extract, distinct
from dotenv import load_dotenv
import os
import pandas as pd
import torch
from train_predict import update_predictions, create_prediction_table
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configuração do CORS para permitir requisições
CORS(app, origins="*")

# Configuração do banco de dados
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo da tabela de products
class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    expiration_date = db.Column(db.Date)
    quantity = db.Column(db.Integer, nullable=False)
    purchase_price = db.Column(db.Numeric(10, 2), nullable=False)
    purchase_currency = db.Column(db.String(10), nullable=False)
    sale_price = db.Column(db.Numeric(10, 2))
    sale_currency = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone.utc))

# Modelo da tabela de predictions
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    categoria = db.Column(db.String(100), nullable=True)
    name = db.Column(db.String(255), nullable=True)  
    ano = db.Column(db.Integer, nullable=False)
    mes = db.Column(db.Integer, nullable=False)
    quantidade = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone.utc))

#Cria as tabelas
with app.app_context():
    db.create_all()

# 1. Obter dados históricos de produtos
@app.route('/api/products', methods=['GET'])
def get_products_data():
    try:
        categoria = request.args.get('category')
        name = request.args.get('name')

        if not categoria and not name:
            return jsonify({'error': 'O parâmetro "category" ou "name" é obrigatório'}), 400
        if categoria and name:
            return jsonify({'error': 'Forneça apenas "category" ou "name", não ambos'}), 400

        query_filter = Product.category == categoria if categoria else Product.name == name
        
        resultados = (db.session.query(
            extract('year', Product.created_at).label('ano'),
            extract('month', Product.created_at).label('mes'),
            func.sum(Product.quantity).label('total_quantidade')
        ).filter(query_filter).group_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        ).order_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        ).all())

        response_key = 'categoria' if categoria else 'name'
        response_value = categoria if categoria else name
        response = {
            response_key: response_value,
            'dados': [
                {
                    'ano': int(resultado.ano),
                    'mes': int(resultado.mes),
                    'quantidade': int(resultado.total_quantidade)
                } for resultado in resultados
            ]
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. Gerar/Atualizar previsões
@app.route('/api/predictions', methods=['POST'])
def update_predictions_endpoint():
    try:
        data = request.get_json()
        categoria = data.get('category')
        name = data.get('name')

        if not categoria and not name:
            return jsonify({'error': 'O campo "category" ou "name" é obrigatório no corpo da requisição'}), 400
        if categoria and name:
            return jsonify({'error': 'Forneça apenas "category" ou "name", não ambos'}), 400
            
        key_type = 'category' if categoria else 'name'
        key_value = categoria if categoria else name
        
        query_filter = Product.category == key_value if key_type == 'category' else Product.name == key_value
        
        # Busca os dados históricos (mesma lógica do endpoint GET /api/products)
        resultados = (db.session.query(
            extract('year', Product.created_at).label('ano'),
            extract('month', Product.created_at).label('mes'),
            func.sum(Product.quantity).label('quantidade')
        ).filter(query_filter).group_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        ).all())
        
        if not resultados:
            return jsonify({'error': f'Nenhum dado encontrado para {key_type} {key_value}'}), 404

        df = pd.DataFrame([{'ano': r.ano, 'mes': r.mes, 'quantidade': r.quantidade} for r in resultados])
        
        create_prediction_table()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Chama a função `update_predictions` atualizada
        predictions = update_predictions(df, key_type, key_value, device)

        return jsonify({
            key_type: key_value,
            'previsoes': predictions
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 3. Obter a lista de categorias
@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        categorias = db.session.query(distinct(Product.category)).all()
        categorias_list = [categoria[0] for categoria in categorias]
        
        if not categorias_list:
            return jsonify({'error': 'Nenhuma categoria encontrada'}), 404

        return jsonify({'categorias': categorias_list}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Consultar as previsões salvas
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        categoria = request.args.get('category')
        name = request.args.get('name')
        
        query = db.session.query(Prediction)
        
        if categoria:
            query = query.filter(Prediction.categoria == categoria)
        if name:
            query = query.filter(Prediction.name == name)
        
        resultados = query.order_by(
            Prediction.categoria,
            Prediction.ano,
            Prediction.mes
        ).all()

        if not resultados:
            return jsonify({'error': 'Nenhuma previsão encontrada'}), 404

        # Agrupar resultados por categoria
        grouped_data = {}
        for resultado in resultados:
        # Define a chave e o tipo (categoria ou nome)
            key_type = 'categoria' if resultado.categoria else 'name'
            key_value = resultado.categoria if resultado.categoria else resultado.name

            if key_value not in grouped_data:
                grouped_data[key_value] = {'key_type': key_type, 'dados': []}
        
            grouped_data[key_value]['dados'].append({
                'ano': resultado.ano,
                'mes': resultado.mes,
                'quantidade': resultado.quantidade
            })

        response = [
            {item['key_type']: key, 'dados': item['dados']}
            for key, item in grouped_data.items()
        ]

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#End points para nome

@app.route('/api/names', methods=['GET'])
def get_names():
    try:
        names = db.session.query(distinct(Product.name)).all()
        names_list = [name[0] for name in names]
        
        if not names_list:
            return jsonify({'error': 'Nenhum nome de produto encontrado'}), 404

        return jsonify({'names': names_list}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)