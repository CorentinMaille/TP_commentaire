from flask import Flask, jsonify, request, abort
from data_service import DataService

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/check-comment', methods=['POST'])
def check_comment(comment):
    # Vérifier si le commentaire est dans la base de données
        # si non, return http NOT_FOUND
        if (not DataService.is_in_db()):
             abort(404)
        else:
            comment_status = DataService.verify_comment(comment)


        return jsonify(status=comment_status)

if __name__ == '__main__':
    app.run(debug=True)
