import pandas as pd
from flask import Blueprint, request, jsonify

from __init__ import db, cache
from ml.sample_recommendation import train_rating_model_with_svd, get_n_popular_movies, \
    get_n_rating_movies, predict_rating_with_svd, get_n_recommended_movies_for_user

recommender = Blueprint('recommender', __name__)

N_TOP_DEFAULT = 10


@recommender.route('/trend/popular', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_popular_movies():
    query = {}
    genres = request.args.get('genres')

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    top = request.args.get('top')
    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return get_n_popular_movies(data, top).to_json(orient='records')


@recommender.route('/trend/rating', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_rating_movies():
    query = {}
    genres = request.args.get('genres')

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    top = request.args.get('top')
    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return get_n_rating_movies(data, top).to_json(orient='records')


@recommender.route('/predict/rating', methods=['GET'])
def get_predicted_rating():
    user_id = request.args.get('userId')
    movie_id = request.args.get('movieId')

    if not user_id or not movie_id:
        return jsonify({'message': 'Missing userId or movieId'})
    else:
        user_id = int(user_id)
        movie_id = int(movie_id)

    return jsonify({'userId': user_id,
                    'movieId': movie_id,
                    'predicted_rating': predict_rating_with_svd(user_id, movie_id)
                    })


@recommender.route('/train/rating', methods=['GET'])
def train_rating_model():
    ratings = list(db.ratings.find({}, {'_id': False}))
    data = pd.DataFrame(ratings)
    best_params, best_score = train_rating_model_with_svd(data)

    return jsonify({'message': 'Done', 'SVD': {'best_params': best_params, 'best_score': best_score}})


@recommender.route('/user/<user_id>', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_recommended_movies(user_id):
    query = {}
    genres = request.args.get('genres')

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    top = request.args.get('top')
    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return get_n_recommended_movies_for_user(int(user_id), top, data).to_json(orient='records')
