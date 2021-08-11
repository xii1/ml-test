import base64
import io

import cv2
import numpy as np
from PIL import Image


def convert_image_to_base64(img):
    img = Image.open(img).convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="png")
    return base64.b64encode(buffered.getvalue()).decode()


def convert_plot_to_base64(plot):
    img = io.BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def convert_opencv_image_to_base64(img):
    _, im_arr = cv2.imencode('.png', img)
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode()


def get_image_with_heatmap_overlay(img, heatmap):
    image = Image.open(img).convert('RGB')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_image = heatmap * 0.8 + image

    return superimposed_image
