from flask import Flask
from api.sample import sample
from api.visualization import visual

app = Flask(__name__)

app.register_blueprint(sample, url_prefix='/sample')
app.register_blueprint(visual, url_prefix='/visual')

if __name__ == "__main__":
    app.run('localhost', 8080)
