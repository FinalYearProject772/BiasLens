#!/usr/bin/env python
import sys
import os

# Add user site-packages to path
appdata = os.environ.get('APPDATA')
user_site_packages = os.path.join(appdata, 'Python', 'Python311', 'site-packages')
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

import streamlit.cli

if __name__ == "__main__":
    streamlit.cli.main(['streamlit', 'run', 'app.py'])