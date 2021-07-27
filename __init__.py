from flask import Flask
from flask_caching import Cache


def create_app():
    flask_app = Flask('app')
    flask_app.config.from_object('config.Config')
    return flask_app


app = create_app()
cache = Cache(app)
