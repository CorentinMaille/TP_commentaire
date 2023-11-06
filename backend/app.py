from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from data_service import DataService

app = Flask(__name__)
CORS(app)

dataservice = DataService()

@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/check-comment', methods=['POST'])
def check_comment():
        comment = request.json.get('comment')
        # Vérifier si le commentaire est négatif ou positif
        status = dataservice.verify_comment(comment)

        return jsonify({'status': status})

if __name__ == '__main__':
    app.run(debug=True)
