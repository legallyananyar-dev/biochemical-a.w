import os
from app import app

if __name__ == '__main__':
    # Dynamically grab the port Render assigns, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' allows external incoming internet traffic to hit your Web UI interface
    app.run(host='0.0.0.0', port=port, debug=False)
