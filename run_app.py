"""
Simple startup script that bypasses configuration issues
"""
import os
import sys

# Unset problematic environment variables
problematic_vars = [
    'KAFKA_BOOTSTRAP_SERVERS',
    'REDIS_PASSWORD',
    'SENTRY_DSN',
    'SMTP_HOST',
    'SMTP_USERNAME',
    'SMTP_PASSWORD',
    'SLACK_WEBHOOK_URL',
    'SSL_CERT_PATH',
    'SSL_KEY_PATH',
]

for var in problematic_vars:
    if var in os.environ and os.environ[var] == '':
        del os.environ[var]

# Now start uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

