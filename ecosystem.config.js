module.exports = {
    apps: [
      {
        name: 'ledovod-ai',
        script: 'venv/bin/gunicorn',
        args: 'app.main:app --bind http://localhost:3000 --workers 3',
        interpreter: 'none',
        env: {
          FLASK_ENV: 'production',
        },
      },
    ],
  };
  