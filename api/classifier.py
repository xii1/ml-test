from flask import Blueprint, render_template, request

from ml.cifar10_mnist import predict_cifar10
from ml.dogcat_classification import predict_dog_cat
from ml.fashion_mnist import predict_fashion
from ml.handwritten_mnist import predict_handwritten
from ml.transfer_learning_classification import predict_image
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


@classifier.route('/handwritten', methods=['GET', 'POST'])
def recognize_handwritten():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        img = convert_image_to_base64(uploaded_file)
        result = predict_handwritten(uploaded_file)
        return render_template('result.html', img=img, message=result)
    return render_template('upload.html', url='handwritten')


@classifier.route('/fashion', methods=['GET', 'POST'])
def recognize_fashion():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        img = convert_image_to_base64(uploaded_file)
        result = predict_fashion(uploaded_file)
        return render_template('result.html', img=img, message=result)
    return render_template('upload.html', url='fashion')


@classifier.route('/cifar10', methods=['GET', 'POST'])
def recognize_cifar10():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        img = convert_image_to_base64(uploaded_file)
        result = predict_cifar10(uploaded_file)
        return render_template('result.html', img=img, message=result)
    return render_template('upload.html', url='cifar10')


@classifier.route('/dogcat_vgg', methods=['GET', 'POST'])
def recognize_dog_cat_with_vgg():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        img = convert_image_to_base64(uploaded_file)
        result = predict_image(uploaded_file)
        return render_template('result.html', img=img, message=result)
    return render_template('upload.html', url='dogcat')
