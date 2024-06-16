module.exports = {
    apps: [
      {
        name: 'ledovod-ai',
        script: 'venv/bin/gunicorn',
        args: 'app:app --bind 0.0.0.0:3000 --workers 3',
        interpreter: 'none',
        env: {
          FLASK_ENV: 'production',
        },
      },
    ],
  };
  