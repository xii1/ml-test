from flask import Blueprint, render_template, request

from ml.sample_classification import predict_dog_cat
from utils import convert_image_to_base64

classifier = Blueprint('classifier', __name__)


@classifier.route('/dogcat', methods=['GET', 'POST'])
def recognize_dog_cat():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        img = convert_image_to_base64(uploaded_file)
        result = predict_dog_cat(uploaded_file)
        return render_template('result.html', img=img, message=result)
    return render_template('upload.html', url='dogcat')
