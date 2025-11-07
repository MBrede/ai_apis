# Docker Deployment Guide

Complete guide for running AI APIs in Docker containers with GPU support and MongoDB.

## 📋 Prerequisites

### Required Software
- **Docker**: 24.0+
- **Docker Compose**: 2.0+
- **NVIDIA Docker Runtime**: For GPU support

## 🚀 Fast Setup with Pre-built Base Image

For faster builds and deployments, you can build ONE base image and push it to Docker Hub. This base image contains system dependencies AND common Python packages shared by all ML APIs.

### What's in the Base Image?

**System dependencies:**
- Python 3.12
- CUDA 13.0.2 + cuDNN (for GPU support)
- uv (fast Python package installer)
- System packages (ffmpeg, git, curl, ca-certificates, etc.)

**Common Python packages** (shared by ALL ML APIs):
- **api-core**: FastAPI, uvicorn, gunicorn, pydantic
- **ml-base**: torch (~2.5 GB), numpy

**NOT included** (installed per-service):
- Service-specific packages (transformers, diffusers, whisper, setfit, etc.)
- Application code

This approach means:
- ✅ **ONE** base image for **ALL** services (no duplication)
- ✅ Heavy common packages (torch, FastAPI) installed once (~5-6 GB base)
- ✅ Each service only installs lightweight service-specific packages
- ✅ Base image changes only when Python/CUDA/torch versions upgrade

### Building and Pushing Base Image

```bash
# Build and push to Docker Hub (requires Docker Hub account)
./scripts/build_and_push_base.sh 1.0.0 yourdockerhubusername

# The script will:
# 1. Build the base image with system deps + common packages (api-core, ml-base)
# 2. Tag it with version and 'latest'
# 3. Push to Docker Hub: yourdockerhubusername/ai-apis-base:1.0.0
# 4. Scan for vulnerabilities (if Trivy is installed)
```

### Using Pre-built Base Image

Once pushed, you can use the faster `.hub` Dockerfiles:

```bash
# Update the FROM line in docker/Dockerfile.*.hub files with your username
sed -i 's/yourusername/actualdockerhubusername/g' docker/Dockerfile.*.hub

# Build services using the pre-built base (much faster!)
docker build -f docker/Dockerfile.stable_diffusion.hub -t ai_apis_sd:latest .
docker build -f docker/Dockerfile.whisper.hub -t ai_apis_whisper:latest .
docker build -f docker/Dockerfile.text_analysis.hub -t ai_apis_text:latest .
docker build -f docker/Dockerfile.bot.hub -t ai_apis_bot:latest .
```

**Build time comparison:**

| Build Type | First Build | Rebuild (code change) |
|------------|-------------|----------------------|
| Without base | ~15-20 min | ~15-20 min |
| With base (initial) | ~3-5 min* | ~2-3 min |

*After base image is pulled once

**Benefits:**
- ⚡ **3-5x faster builds**: Skip reinstalling system packages
- 🔄 **Consistency**: All services use identical system environment
- 💾 **Efficient caching**: System deps cached once, Python packages per-service
- 🚀 **CI/CD friendly**: Fast iterations, clear separation of concerns

**When to rebuild base image:**
- Python version upgrade
- CUDA version change
- PyTorch version upgrade
- FastAPI or other api-core package major updates
- System package updates (ffmpeg, git, etc.)
- uv version upgrade

Typically once every few months, not on every code change.

**Package installation breakdown:**
```
Base image (~5-6 GB):
  - System: Python, CUDA, uv, ffmpeg, git, etc.
  - api-core: FastAPI, uvicorn, gunicorn, pydantic (~100 MB)
  - ml-base: torch (~2.5 GB), numpy (~50 MB)

Per-service (additional):
  - stable-diffusion-only: diffusers, transformers, accelerate, etc. (~2-3 GB)
  - whisper-only: openai-whisper, pyannote-audio (~500 MB)
  - text-analysis-only: transformers, setfit, huggingface-hub (~1-2 GB)
```

### Install NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 Quick Start

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Required settings:**
```bash
# API Keys (CHANGE THESE!)
API_KEY=your-secure-api-key-here
ADMIN_API_KEY=your-secure-admin-key-here

# HuggingFace token
HF_TOKEN=hf_your_token_here

# MongoDB credentials
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=your-secure-password

# MongoDB authentication
USE_MONGODB=true
MONGODB_URL=mongodb://admin:your-secure-password@mongodb:27017/
MONGODB_DB=ai_apis
```

### 2. Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 3. Initialize MongoDB

```bash
# Wait for MongoDB to be ready (about 10 seconds)
sleep 10

# Initialize database with API keys
docker-compose exec mongodb mongosh -u admin -p your-secure-password --authenticationDatabase admin ai_apis --eval "
db.api_keys.insertOne({
    key: 'your-secure-api-key-here',
    name: 'Default API Key',
    is_admin: false,
    active: true,
    created_at: new Date(),
    rate_limit: 1000,
    usage_count: 0
});
db.api_keys.insertOne({
    key: 'your-secure-admin-key-here',
    name: 'Admin API Key',
    is_admin: true,
    active: true,
    created_at: new Date(),
    rate_limit: 10000,
    usage_count: 0
});
"

# Or use the Python script
docker-compose exec stable_diffusion python /app/scripts/init_mongodb.py
```

### 4. Test APIs

```bash
# Test Stable Diffusion
curl -H "X-API-Key: your-secure-api-key-here" \
    http://localhost:1234/get_available_stable_diffs

# Test Whisper
curl -H "X-API-Key: your-secure-api-key-here" \
    -F "file=@test.wav" \
    http://localhost:8080/transcribe/

# Test Sentiment Analysis
curl -H "X-API-Key: your-secure-api-key-here" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"text":["This is amazing!"], "model":"oliverguhr/german-sentiment-bert"}' \
    http://localhost:8001/predict_sentiment/
```

## 📦 Services Overview

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| **mongodb** | 27017 | ❌ | Authentication & settings storage |
| **stable_diffusion** | 1234 | ✅ | Image generation with LORA support |
| **whisper** | 8080 | ✅ | Audio transcription + diarization |
| **sentiment** | 8001 | ✅ | Sentiment analysis |
| **text_classification** | 8002 | ✅ | Text classification |
| **telegram_bot** | - | ❌ | Telegram bot interface |

## 🔧 Service Management

### Start/Stop Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d stable_diffusion

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data!)
docker-compose down -v

# Restart service
docker-compose restart whisper
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f stable_diffusion

# Last 100 lines
docker-compose logs --tail=100 whisper
```

### Access Containers

```bash
# Open shell in container
docker-compose exec stable_diffusion bash

# Run command in container
docker-compose exec whisper python -c "import torch; print(torch.cuda.is_available())"

# Check GPU usage
docker-compose exec stable_diffusion nvidia-smi
```

## 💾 Data Persistence

### Volumes

Docker volumes persist data across container restarts:

```bash
# List volumes
docker volume ls | grep ai_apis

# Inspect volume
docker volume inspect ai_apis_mongodb_data

# Backup MongoDB
docker-compose exec mongodb mongodump --archive=/backup.gz --gzip
docker cp ai_apis_mongodb:/backup.gz ./mongodb_backup.gz

# Restore MongoDB
docker cp ./mongodb_backup.gz ai_apis_mongodb:/backup.gz
docker-compose exec mongodb mongorestore --archive=/backup.gz --gzip
```

### Host Directories

Mounted directories (defined in docker-compose.yml):
- `./loras` → SD LORA models
- `./models` → Downloaded models
- `*_cache` volumes → HuggingFace cache

## 🔐 MongoDB Management

### Access MongoDB Shell

```bash
# Using mongosh
docker-compose exec mongodb mongosh -u admin -p your-secure-password --authenticationDatabase admin

# Or using docker exec
docker exec -it ai_apis_mongodb mongosh -u admin -p your-secure-password
```

### Manage API Keys

```javascript
// Connect to database
use ai_apis

// List all API keys
db.api_keys.find().pretty()

// Add new API key
db.api_keys.insertOne({
    key: "new-api-key-123",
    name: "Client A",
    is_admin: false,
    active: true,
    created_at: new Date(),
    rate_limit: 500,
    usage_count: 0
})

// Deactivate key
db.api_keys.updateOne(
    {key: "old-api-key"},
    {$set: {active: false}}
)

// Check usage
db.api_keys.find({}, {name: 1, usage_count: 1, rate_limit: 1})

// View usage logs
db.usage_logs.find().sort({timestamp: -1}).limit(10).pretty()
```

### Manage Bot Settings

```javascript
// Get user settings
db.bot_settings.findOne({user_id: "123456789"})

// Update SD parameters for user
db.bot_settings.updateOne(
    {user_id: "123456789"},
    {
        $set: {
            "sd_parameters.steps": 50,
            "sd_parameters.guidance_scale": 8.5,
            "updated_at": new Date()
        }
    },
    {upsert: true}
)

// List all users
db.bot_settings.find({}, {user_id: 1, llm_mode: 1})
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check docker-compose GPU config
docker-compose config | grep -A 5 "nvidia"

# Check container GPU access
docker-compose exec stable_diffusion nvidia-smi
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Restart service to clear cache
docker-compose restart stable_diffusion

# Adjust buffer timeouts in .env
# Shorter timeouts = more aggressive unloading
```

### MongoDB Connection Issues

```bash
# Check MongoDB status
docker-compose logs mongodb

# Test connection
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Verify credentials
docker-compose exec mongodb mongosh -u admin -p your-password --eval "db.version()"
```

### Service Won't Start

```bash
# Check logs
docker-compose logs service_name

# Rebuild image
docker-compose build --no-cache service_name

# Check configuration
docker-compose config

# Verify .env file
cat .env | grep -v "^#"
```


## 📊 Monitoring

### Check Service Health

```bash
# Docker health checks
docker-compose ps

# API endpoints
curl http://localhost:1234/docs  # Stable Diffusion
curl http://localhost:8080/docs  # Whisper
curl http://localhost:8001/docs  # Sentiment

# MongoDB health
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
```

### Resource Usage

```bash
# Container stats
docker stats

# GPU usage
watch -n 1 nvidia-smi

# Disk usage
docker system df
```

## 🔒 Security Best Practices

1. **Change Default Passwords**
   ```bash
   # Generate secure passwords
   openssl rand -base64 32
   ```

2. **Use Secrets Management**
   ```bash
   # Don't commit .env to git
   echo ".env" >> .gitignore

   # Use Docker secrets in production
   docker secret create api_key ./api_key.txt
   ```

3. **Network Isolation**
   ```yaml
   # In docker-compose.yml
   networks:
     ai_apis_network:
       internal: true  # No external access
   ```

4. **Limit Resources**
   ```yaml
   # Add to service in docker-compose.yml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 16G
   ```

5. **Enable MongoDB Authentication**
   - Always set `MONGO_ROOT_USER` and `MONGO_ROOT_PASSWORD`
   - Never expose port 27017 to the internet
   - Use TLS for production deployments

## 🚀 Production Deployment

### Use docker-compose.prod.yml

```yaml
version: '3.8'

services:
  stable_diffusion:
    image: your-registry.com/ai_apis_sd:latest
    restart: always
    environment:
      - REQUIRE_AUTH=true
      - USE_MONGODB=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 24G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:1234/docs"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Deploy

```bash
# Build and push images
docker-compose build
docker tag ai_apis_stable_diffusion your-registry.com/ai_apis_sd:latest
docker push your-registry.com/ai_apis_sd:latest

# Deploy on production server
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📚 Additional Resources

- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
- [MongoDB Docker Guide](https://hub.docker.com/_/mongo)
- [FastAPI Docker Documentation](https://fastapi.tiangolo.com/deployment/docker/)
