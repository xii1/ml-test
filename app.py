from flask import Flask

from api.classifier import classifier
from api.sample import sample
from api.visualization import visual

app = Flask(__name__)

app.register_blueprint(sample, url_prefix='/sample')
app.register_blueprint(visual, url_prefix='/visual')
app.register_blueprint(classifier, url_prefix='/classifier')

if __name__ == "__main__":
    app.run('localhost', 8080)
