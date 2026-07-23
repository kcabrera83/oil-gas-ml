# Deployment Guide - Crude Oil Evaluation

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/train.py

EXPOSE 5001

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t oil-gas-ml .
docker run -p 5001:5001 oil-gas-ml
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5001:5001"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5001 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
# Clone repository
git clone https://github.com/kcabrera83/oil-gas-ml.git
cd oil-gas-ml

# Install dependencies
pip install -r requirements.txt

# Train models
python scripts/train.py

# Run evaluation (optional)
python scripts/evaluate.py

# Start server
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy (nginx/Apache)
- Add rate limiting to prediction endpoints
- Validate input ranges before model inference

### Monitoring
- Monitor `/api/health` endpoint for uptime
- Log all prediction requests for auditing
- Track model prediction distributions for drift detection

### Performance
- Pre-load models at startup (handled in `app.py`)
- Use connection pooling for any database integrations
- Consider model caching for high-throughput scenarios

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5001/api/docs`
