
from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
from apps.data.models import SoilData, WeatherData
from apps.crop.models import Location
from apps.model.models import Prediction

@blueprint.route('/index')
@login_required
def index():
    total_locations = Location.query.count()
    total_predictions = Prediction.query.count()
    total_soil_records = SoilData.query.count()
    total_weather_records = WeatherData.query.count()
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()

    from collections import Counter
    crop_counts = Counter([p.crop_recommended for p in Prediction.query.all()])
    labels = list(crop_counts.keys())
    data = list(crop_counts.values())

    return render_template(
        'home/index.html',
        segment='index',
        total_locations=total_locations,
        total_predictions=total_predictions,
        total_soil_records=total_soil_records,
        total_weather_records=total_weather_records,
        recent_predictions=recent_predictions,
        labels=labels,
        data=data
    )



@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
