# Docker Setup for ResuMatch AI

This guide explains how to run ResuMatch AI using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t resumatch-ai .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name resumatch-ai \
     -p 5000:5000 \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/storage:/app/storage \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/results:/app/results \
     resumatch-ai
   ```

3. **View logs:**
   ```bash
   docker logs -f resumatch-ai
   ```

4. **Stop the container:**
   ```bash
   docker stop resumatch-ai
   docker rm resumatch-ai
   ```

## Environment Variables

You can set environment variables in `docker-compose.yml` or pass them when running:

```bash
docker run -d \
  -p 5000:5000 \
  -e ENABLE_OPENAI=1 \
  -e OPENAI_API_KEY=your_key_here \
  resumatch-ai
```

### Available Environment Variables:

- `ENABLE_OPENAI`: Set to `1` to enable OpenAI features (default: `0`)
- `OPENAI_API_KEY`: Your OpenAI API key (required if ENABLE_OPENAI=1)
- `FLASK_ENV`: Set to `production` or `development` (default: `production`)
- `DEBUG`: Set to `1` to enable debug mode (default: `0`)

## Accessing the Application

Once the container is running, access the application at:
- **http://localhost:5000**

## Data Persistence

The following directories are mounted as volumes to persist data:
- `uploads/` - Uploaded resume files
- `storage/profiles/` - User profile data
- `logs/` - Application logs
- `results/` - Analysis results

## Troubleshooting

### Container won't start

1. **Check logs:**
   ```bash
   docker-compose logs resumatch
   ```

2. **Verify the image was built correctly:**
   ```bash
   docker images | grep resumatch
   ```

3. **Check if port 5000 is already in use:**
   ```bash
   # Linux/Mac
   lsof -i :5000
   
   # Windows
   netstat -ano | findstr :5000
   ```

### Model initialization fails

- Ensure `data/sample_internships.json` exists in the project directory
- Check that all Python dependencies installed correctly

### OCR not working

- Tesseract OCR is installed in the container
- If issues persist, check the logs for OCR-related errors

## Building for Production

For production deployment, consider:

1. **Use a production WSGI server:**
   ```dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
   ```

2. **Set proper environment variables:**
   - `FLASK_ENV=production`
   - `DEBUG=0`

3. **Use a reverse proxy** (nginx, Traefik, etc.) in front of the container

## Updating the Application

1. **Stop the container:**
   ```bash
   docker-compose down
   ```

2. **Rebuild the image:**
   ```bash
   docker-compose build --no-cache
   ```

3. **Start again:**
   ```bash
   docker-compose up -d
   ```

## Health Check

The container includes a health check endpoint at `/health`. Docker will automatically monitor this.

Check health status:
```bash
docker ps  # Look for "healthy" status
```

