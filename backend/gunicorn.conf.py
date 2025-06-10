import multiprocessing
import os

# Gunicorn configuration
port = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Optimize worker settings for memory
workers = 2  # Reduced number of workers
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Memory optimization
max_requests = 1000
max_requests_jitter = 50
worker_tmp_dir = "/dev/shm"  # Use RAM for temporary files

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "resume-refiner-backend"

# SSL (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile" 