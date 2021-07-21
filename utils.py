import base64
import io


def convert_image_to_base64(img):
    data = img.getvalue()
    data = base64.b64encode(data)
    return data.decode()


def convert_plot_to_base64(pyplot):
    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    return convert_image_to_base64(img)
