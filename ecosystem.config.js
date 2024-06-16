module.exports = {
  apps: [
    {
      name: "ledovod-ai",
      script: "venv/bin/gunicorn",
      args: "app.main:app --bind localhost:3000 --workers 3 --worker-class uvicorn.workers.UvicornWorker",
      interpreter: "none",
      env: {
        FLASK_ENV: "production",
      },
    },
  ],
};
