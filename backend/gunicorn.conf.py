import multiprocessing
import os

# Gunicorn configuration
port = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{port}"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "resume-refiner-backend"

# SSL (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile" 