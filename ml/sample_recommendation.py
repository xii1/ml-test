QUANTILES_THRESHOLD = 0.95


def get_popular_movies(data, size):
    return data.nlargest(size, 'popularity')[['id', 'original_title', 'genres', 'popularity']]


def get_rating_movies(data, size):
    m = data['vote_count'].quantile(QUANTILES_THRESHOLD)
    c = data['vote_average'].mean()

    rating_movies = data.copy().loc[data['vote_count'] >= m]
    rating_movies['rating_score'] = rating_movies.apply(lambda movie: calc_weighted_rating(movie, m, c), axis=1)

    return rating_movies.nlargest(size, 'rating_score')[['id', 'original_title', 'genres', 'vote_count', 'vote_average', 'rating_score']]


def calc_weighted_rating(movie, m, c):
    v = movie['vote_count']
    r = movie['vote_average']
    return (v * r + m * c) / (v + m)
