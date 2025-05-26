from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

# Inicializar Flask y Flask-RESTx
app = Flask(__name__)
api = Api(app, version='1.0', title='API de Predicción de Géneros de Películas',
          description='Predice géneros a partir de la sinopsis')

ns = api.namespace('predict', description='Operaciones de predicción')

# Modelo de entrada: solo el texto de la sinopsis
input_model = api.model('Input', {
    'plot': fields.String(required=True, description='Sinopsis de la película')
})

# Intenta cargar el modelo USE local
try:
    USE_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "use_model")
    embed = hub.load(USE_LOCAL_PATH)
    print("✅ Modelo USE cargado correctamente")
except Exception as e:
    print("⚠️ Modelo USE no cargado:", e)
    embed = None

# Cargar modelo de predicción
try:
    model_path = os.path.join(os.path.dirname(__file__), "modelo_generos.keras")
    clf = load_model(model_path)
    print("✅ Modelo de géneros cargado correctamente")
except Exception as e:
    print("⚠️ Modelo de géneros no cargado:", e)
    clf = None

# Etiquetas de géneros esperadas
GENRES = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
          'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

@ns.route('/')
class GenrePredictor(Resource):
    @ns.expect(input_model)
    def post(self):
        if not embed or not clf:
            return {"error": "Modelos no cargados"}, 500

        try:
            plot = api.payload['plot']
            print(f"📥 Sinopsis recibida: {plot}")
            embedding = embed([plot]).numpy()
            prediction = clf.predict(embedding)[0]
            resultado = {genre: float(score) for genre, score in zip(GENRES, prediction)}
            return {"predicted_genres": resultado}
        except Exception as e:
            print("❌ Error durante la predicción:", e)
            return {"error": str(e)}, 500

    @ns.doc(params={'plot': 'Sinopsis de la película'})
    def get(self):
        if not embed or not clf:
            return {"error": "Modelos no cargados"}, 500

        try:
            plot = request.args.get("plot")
            if not plot:
                return {"error": "Falta el parámetro 'plot'"}, 400

            print(f"📥 Sinopsis recibida (GET): {plot}")
            embedding = embed([plot]).numpy()
            prediction = clf.predict(embedding)[0]
            resultado = {genre: float(score) for genre, score in zip(GENRES, prediction)}
            return {"predicted_genres": resultado}
        except Exception as e:
            print("❌ Error durante la predicción (GET):", e)
            return {"error": str(e)}, 500

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
