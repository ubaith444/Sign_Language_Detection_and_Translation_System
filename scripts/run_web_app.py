import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import WEB_HOST, WEB_PORT
from src.web.app import create_app

def main():
    print("=================================================================")
    print(" STARTING ISL WEB INTERFACE SERVER                               ")
    print(f" URL: http://{WEB_HOST}:{WEB_PORT}                              ")
    print("=================================================================")
    app = create_app()
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
