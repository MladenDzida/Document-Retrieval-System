#!/bin/bash

# Start only Jupyter server
#jupyter notebook --ip=0.0.0.0 --port=8080 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

# start only llm_app
#streamlit run llm_app.py --server.port=8501 --server.address=0.0.0.0

# start both
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf