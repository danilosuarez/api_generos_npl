from flask import Flask
from flask_restx import Api, Resource, fields
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo de géneros
try:
    model = load_model('modelo_generos.keras')
    print("✅ Modelo cargado correctamente")
except:
    print("⚠️ Modelo no cargado.")
    model = None

# Cargar el modelo de Universal Sentence Encoder
try:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("✅ USE cargado correctamente")
except:
    print("⚠️ Modelo USE no cargado.")
    embed = None

# Inicializar la app
app = Flask(__name__)
api = Api(app, version='1.0', title='Predicción de Géneros de Películas',
          description='API que predice la probabilidad de géneros basada en el plot (descripción)')

ns = api.namespace('predict', description='Predicción de géneros')

# Parámetros esperados
parser = ns.parser()
parser.add_argument('plot', type=str, required=True, help='Descripción de la película', location='args')

# Salida esperada
resource_fields = api.model('Prediction', {
    'predicciones': fields.List(fields.Float),
})

@ns.route('/')
class GenrePredictor(Resource):

    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']

        if model is None or embed is None:
            return {'predicciones': [-1.0] * 24}, 200

        # Generar embedding de la descripción
        embedding = embed([plot]).numpy()
        
        # Hacer predicción
        pred = model.predict(embedding)[0].tolist()
        
        return {'predicciones': pred}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
