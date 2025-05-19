from flask import Blueprint, render_template

blueprint = Blueprint('model_blueprint', __name__, url_prefix='/model')

@blueprint.route('/')
def prediction():
    return render_template('predictions/view_predictions.html')
