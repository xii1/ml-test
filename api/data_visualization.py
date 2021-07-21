from flask import Blueprint, render_template

from ml.sample_visualization import process_sample_data, process_csv_data

dv = Blueprint('dv', __name__)


@dv.route('/sample', methods=['GET'])
def sample():
    img = process_sample_data()
    return render_template('chart.html', img=img)


@dv.route('/water', methods=['GET'])
def water():
    img = process_csv_data()
    return render_template('chart.html', img=img)
