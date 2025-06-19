from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import func, extract, distinct
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd
import torch
from train-predict import update_predictions, create_prediction_table

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configuração do CORS para permitir requisições
CORS(app, resources={r"/api/*": {"origins": "http://localhost:9000"}})

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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Modelo da tabela de predictions
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    categoria = db.Column(db.String(100), nullable=False)
    ano = db.Column(db.Integer, nullable=False)
    mes = db.Column(db.Integer, nullable=False)
    quantidade = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@app.route('/api/products_por_categoria', methods=['GET'])
def get_products_por_categoria():
    try:
        categoria = request.args.get('categoria')
        if not categoria:
            return jsonify({'error': 'Categoria é obrigatória'}), 400

        resultados = (db.session.query(
            extract('year', Product.created_at).label('ano'),
            extract('month', Product.created_at).label('mes'),
            func.sum(Product.quantity).label('total_quantidade')
        )
        .filter(Product.category == categoria)
        .group_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        )
        .order_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        )
        .all())

        response = {
            'categoria': categoria,
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

@app.route('/api/atualizar_previsoes', methods=['POST'])
def atualizar_previsoes():
    try:
        categoria = request.args.get('categoria')
        if not categoria:
            return jsonify({'error': 'Categoria é obrigatória'}), 400

        resultados = (db.session.query(
            extract('year', Product.created_at).label('ano'),
            extract('month', Product.created_at).label('mes'),
            func.sum(Product.quantity).label('quantidade')
        )
        .filter(Product.category == categoria)
        .group_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        )
        .order_by(
            extract('year', Product.created_at),
            extract('month', Product.created_at)
        )
        .all())

        if not resultados:
            return jsonify({'error': f'Nenhum dado encontrado para a categoria {categoria}'}), 404

        df = pd.DataFrame([
            {'ano': int(r.ano), 'mes': int(r.mes), 'quantidade': int(r.quantidade)}
            for r in resultados
        ])

        create_prediction_table()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictions = update_predictions(df, categoria, device)

        return jsonify({
            'categoria': categoria,
            'previsoes': predictions
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/categorias', methods=['GET'])
def get_categorias():
    try:
        categorias = db.session.query(distinct(Product.category)).all()
        categorias_list = [categoria[0] for categoria in categorias]
        
        if not categorias_list:
            return jsonify({'error': 'Nenhuma categoria encontrada'}), 404

        return jsonify({'categorias': categorias_list}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/previsoes', methods=['GET'])
def get_previsoes():
    try:
        categoria = request.args.get('categoria')
        
        query = db.session.query(
            Prediction.categoria,
            Prediction.ano,
            Prediction.mes,
            Prediction.quantidade
        )
        
        if categoria:
            query = query.filter(Prediction.categoria == categoria)
        
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
            cat = resultado.categoria
            if cat not in grouped_data:
                grouped_data[cat] = []
            grouped_data[cat].append({
                'ano': resultado.ano,
                'mes': resultado.mes,
                'quantidade': resultado.quantidade
            })

        response = [
            {'categoria': cat, 'dados': dados}
            for cat, dados in grouped_data.items()
        ]

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)