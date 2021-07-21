from flask import Blueprint, jsonify, render_template

sample = Blueprint('sample', __name__)


@sample.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello'})


@sample.route('/', methods=['GET'])
def home():
    return render_template('index.html')
