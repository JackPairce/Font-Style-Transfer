from flask import Blueprint

bp = Blueprint("hello", __name__)


@bp.route("/")
def hello():
    return "Hello to Font Style Transfer API"
