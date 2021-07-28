from __init__ import cache, db

import random
from flask import Blueprint, jsonify, render_template


sample = Blueprint('sample', __name__)


@sample.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello'})


@sample.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@sample.route('/random', methods=['GET'])
@cache.cached(timeout=300, key_prefix='cache_rand')
def get_random():
    return jsonify({'Number': random.randint(0, 100000)})


@sample.route('/data', methods=['GET'])
def get_data():
    return jsonify({'Count': db.students.count()})
