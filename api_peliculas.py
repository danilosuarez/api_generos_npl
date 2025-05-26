from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import numpy as np

app = Flask(__name__)
api = Api(app, version='1.0', title='API Predicción Géneros de Películas',
          description='Predice probabilidades de géneros a partir de descripciones de películas.')

ns = api.namespace('predict', description='Predicción de géneros')

parser = ns.parser()
parser.add_argument('plot', type=str, required=True, help='Sinopsis de la película', location='args')

resource_fields = api.model('Resource', {
    'probabilidades': fields.Raw
})

# Cargar modelo
try:
    clf = joblib.load("clf.pkl")
except:
    clf = None
    print("⚠️ Modelo no cargado.")

@ns.route('/')
class GenreClassifier(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']

        # Aquí irá la generación de embeddings cuando esté listo
        if clf:
            dummy_input = np.zeros((1, 512))
            result = clf.predict_proba(dummy_input)
            return {'probabilidades': result.tolist()}
        else:
            return {'probabilidades': 'Modelo no disponible'}, 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
