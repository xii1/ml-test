import pandas as pd
from flask import Blueprint, request, jsonify

from ml.sample_recommendation import get_popular_movies, get_rating_movies
from __init__ import db, cache

recommender = Blueprint('recommender', __name__)

N_TOP_DEFAULT = 10


@recommender.route('/popular', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_popular_movies():
    query = {}
    genres = request.args.get('genres')
    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    top = request.args.get('top')
    if top is None:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)
    if data.size == 0:
        return jsonify([])
    return get_popular_movies(data, top).to_json(orient='records')


@recommender.route('/rating', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_rating_movies():
    query = {}
    genres = request.args.get('genres')
    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    top = request.args.get('top')
    if top is None:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)
    if data.size == 0:
        return jsonify([])
    return get_rating_movies(data, top).to_json(orient='records')
