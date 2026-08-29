from flask import Flask
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
import app.routes # Hooks your endpoint architecture directly into the thread
if __name__ == '__main__':
    app.run(port=5000)
