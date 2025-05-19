from flask import Blueprint, render_template

blueprint = Blueprint('user_blueprint', __name__, url_prefix='/users')

@blueprint.route('/')
def users():
    return render_template('users/view_users.html')
