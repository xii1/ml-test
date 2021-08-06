import base64
import io

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
