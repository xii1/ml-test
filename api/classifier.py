from flask import Blueprint, render_template, request

from utils import convert_image_to_base64

classifier = Blueprint('classifier', __name__)


@classifier.route('/dogcat', methods=['GET', 'POST'])
def recognize_dog_cat():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        img = convert_image_to_base64(uploaded_file)
        return render_template('result.html', img=img, message='Dog')
    return render_template('upload.html', url='dogcat')
