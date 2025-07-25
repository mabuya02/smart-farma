from flask import Blueprint

blueprint = Blueprint(
    'model_blueprint',
    __name__,
    url_prefix='/predictions'
)

from . import routes
