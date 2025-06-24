from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
app.secret_key = "supersecretkey"
socketio = SocketIO(app)

from app.routes import bp
app.register_blueprint(bp)
