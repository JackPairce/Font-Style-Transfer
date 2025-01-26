from flask import Flask
from flask_cors import CORS
from .routes import hello, service

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "http://localhost:3000",
            "methods": ["POST", "GET"],
        }
    },
)

app.register_blueprint(hello.bp)
# app.register_blueprint(converter.bp)
app.register_blueprint(service.bp)

if __name__ == "__main__":
    app.run()
