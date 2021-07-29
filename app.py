from __init__ import app

from api.caching import caching
from api.classifier import classifier
from api.recommender import recommender
from api.sample import sample
from api.visualization import visual

app.register_blueprint(caching, url_prefix='/caching')
app.register_blueprint(sample, url_prefix='/sample')
app.register_blueprint(visual, url_prefix='/visual')
app.register_blueprint(classifier, url_prefix='/classifier')
app.register_blueprint(recommender, url_prefix='/recommender')

if __name__ == "__main__":
    app.run('localhost', 8080)
