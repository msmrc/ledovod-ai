import multiprocessing

workers = multiprocessing.cpu_count() * 2 + 1
bind = "http://localhost:3000"
worker_class = "uvicorn.workers.UvicornWorker"
