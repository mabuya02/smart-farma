from flask import Blueprint

blueprint = Blueprint(
    'analytics_blueprint',
    __name__,
    url_prefix='/analytics'
)

from . import routes
