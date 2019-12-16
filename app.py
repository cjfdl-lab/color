import os
import uuid

from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug import secure_filename

from model import Net
from torchvision.transforms import ToPILImage


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(CURRENT_DIR, 'static', 'upload_images')
PROCESSED_DIR = os.path.join(CURRENT_DIR, 'static', 'processed_images')
MODEL_PKL = os.path.join(CURRENT_DIR, 'new_model.pth')


DEVICE = 'cpu'  # cuda:0
model_eval = Net(MODEL_PKL).to(DEVICE)


app = Flask(__name__)
app.secret_key = '2b03a51984ae180e'


def hash_filename(filename):
    _, _, suffix = filename.rpartition('.')
    return '%s.%s' % (uuid.uuid4().hex, suffix)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image_origin')
        filename = secure_filename(file.filename)
        hashed_filename = hash_filename(filename)
        file_path = os.path.join(UPLOAD_DIR, hashed_filename)
        file.save(file_path)
        session['last_uploaded_image'] = hashed_filename

        # TODO: extract the convert process as instance method
        img_array = (model_eval.forward(file_path) * 255).astype('uint8')
        proceesed = ToPILImage()(img_array)
        proceesed_image = 'processed_' + hashed_filename
        proceesed.save(os.path.join(PROCESSED_DIR, proceesed_image))
        session['processed_image'] = proceesed_image
        return redirect(url_for('index'))
    return render_template(
        'index.html',
        last_uploaded_image=session.pop('last_uploaded_image', ''),
        processed_image=session.pop('processed_image', '')
    )
