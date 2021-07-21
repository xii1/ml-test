from flask import Flask
from api.sample import sample
from api.data_visualization import dv

app = Flask(__name__)

app.register_blueprint(sample, url_prefix='/sample')
app.register_blueprint(dv, url_prefix='/dv')

if __name__ == "__main__":
    app.run('localhost', 8080)
