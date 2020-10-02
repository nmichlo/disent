import base64
import io
import os
from uuid import uuid4
from collections import defaultdict
import imageio

import flask
from flask import request
import requests

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output


# ========================================================================= #
# OPTIONS                                                                   #
# ========================================================================= #


PORT = 7777
IMAGE_FORMAT = 'png'
REFRESH_MS = 1000
ROW_HEIGHT = 128
ROW_SIZE = 9


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def send_images(images, address='localhost', format=IMAGE_FORMAT):
    return requests.post(f'http://{address}:{PORT}/api/image', files={
        key: imageio.imwrite('<bytes>', image, format=format)
        for key, image in images.items()
    })


def clear_all_images(address='localhost'):
    try:
        return requests.delete(f'http://{address}:{PORT}/api/images')
    except:
        pass


# ========================================================================= #
# SERVER                                                                    #
# ========================================================================= #


if __name__ == '__main__':

    # app & dashboard
    server = flask.Flask(__name__)
    app = dash.Dash(
        __name__,
        server=server,
        routes_pathname_prefix='/',
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
    )

    # data storage
    UUID_TO_IMAGE = {}
    IMAGE_QUEUE = defaultdict(lambda: {
        'pos': 0,
        'items': [None] * ROW_SIZE
    })

    # helper function to store an image in the database, and generate and html component
    def push_list_image(key, img, serve=True):
        uuid = str(uuid4())
        img = imageio.imwrite('<bytes>', img, IMAGE_FORMAT)

        if serve:
            UUID_TO_IMAGE[uuid] = img
            src = f'/api/image/{uuid}'
        else:
            img = base64.b64encode(img).decode()
            src = f'data:image/{IMAGE_FORMAT};base64,{img}'

        # ADD ITEM TO LIST AT CORRECT POSITION
        items, idx = IMAGE_QUEUE[key]['items'], IMAGE_QUEUE[key]['pos']
        IMAGE_QUEUE[key]['pos'] = (idx + 1) % ROW_SIZE
        replaced, items[idx] = items[idx], dict(
            uuid=uuid,
            elem=html.Img(src=src, className='row-image', style={
                'height': f'{ROW_HEIGHT}px',
                'width': 'auto',
            })
        )

        # remove uuid from UUID_TO_IMAGE for removed element
        if replaced is not None:
            if replaced['uuid'] in UUID_TO_IMAGE:
                del UUID_TO_IMAGE[replaced['uuid']]

    # upload images to the database
    @server.route('/api/image', methods=['POST'])
    def upload_image():
        if request.method == 'POST':
            if not request.files:
                return '[ERROR] no media in request'
            for key, file in request.files.items():
                try:
                    img = imageio.imread(file.read())
                    push_list_image(key, img)
                except:
                    pass
            return '[SUCCESS]'

    # retrieve a specific image in the database
    @server.route('/api/image/<uuid>', methods=['GET'])
    def get_image(uuid):
        try:
            if uuid in UUID_TO_IMAGE:
                fp = io.BytesIO(UUID_TO_IMAGE[uuid])
                return flask.send_file(fp, mimetype='image/jpeg') #, cache_timeout=REFRESH_MS/1000*ROW_SIZE*2)
            return 'MISSING'
        except Exception as e:
            return 'ERROR'

    # retrieve a specific image in the database
    @server.route('/api/images', methods=['DELETE'])
    def clear_images():
        IMAGE_QUEUE.clear()
        UUID_TO_IMAGE.clear()
        return 'SUCCESS'

    # refresh with new content
    @app.callback(Output('image-list', 'children'), [Input('interval-component', 'n_intervals')])
    def update_metrics(n):
        rows = []
        # append rows
        for name in IMAGE_QUEUE.keys():
            rows.append(html.Plaintext(name, className='row-heading'))
            rows.append(html.Div([img['elem'] for img in IMAGE_QUEUE[name]['items'] if (img is not None)]))
        # display
        return rows if rows else html.Plaintext('No images uploaded!', className='no-images-message')

    # layout of the home page
    app.layout = html.Div([
        html.Div(id='image-list'),
        dcc.Interval(id='interval-component', interval=REFRESH_MS+1)
    ])

    # start the server!!!
    server.run(debug=False, host='0.0.0.0', port=PORT)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
