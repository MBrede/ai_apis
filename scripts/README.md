# Scripts

Utility scripts for building, deploying, and managing the AI APIs project.

## build_and_push_base.sh

Builds and pushes the base Docker image to Docker Hub. This base image contains all common dependencies and significantly speeds up subsequent service builds.

### Prerequisites

- Docker installed and running
- Docker Hub account
- Logged in to Docker Hub (`docker login`)

### Usage

```bash
./scripts/build_and_push_base.sh [VERSION] [DOCKERHUB_USERNAME]
```

**Parameters:**
- `VERSION`: Image version tag (default: "latest")
- `DOCKERHUB_USERNAME`: Your Docker Hub username (can also be set via environment variable)

**Examples:**

```bash
# Build and push with version 1.0.0
./scripts/build_and_push_base.sh 1.0.0 myusername

# Build and push as latest using environment variable
export DOCKERHUB_USERNAME=myusername
./scripts/build_and_push_base.sh

# Build specific version
./scripts/build_and_push_base.sh 1.2.3 myusername
```

### What It Does

1. **Validates** Docker and Docker Hub login status
2. **Builds** the base image from `docker/Dockerfile.base`
3. **Tags** with both version and 'latest' tags
4. **Pushes** to Docker Hub
5. **Scans** for vulnerabilities (if Trivy is installed)

### Output Image

The script creates and pushes:
- `yourusername/ai-apis-base:VERSION`
- `yourusername/ai-apis-base:latest`

### Benefits

**Build Time Comparison:**

| Build Type | Time | Use Case |
|------------|------|----------|
| From scratch | ~10-15 min | First build, major updates |
| From base image | ~2-3 min | Service updates, code changes |

**Storage Savings:**
- Base image: ~8-10 GB (built once)
- Service images: ~500 MB each (just code + service deps)

### Using the Base Image

After pushing the base image, update your service Dockerfiles:

```dockerfile
# Instead of:
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
# Install everything...

# Use:
FROM myusername/ai-apis-base:latest
# Only copy and configure your service
```

Or use the provided `.hub` Dockerfiles in the `docker/` directory:
- `Dockerfile.stable_diffusion.hub`
- `Dockerfile.whisper.hub`
- `Dockerfile.text_analysis.hub`

### Security Scanning

If [Trivy](https://github.com/aquasecurity/trivy) is installed, the script automatically scans the built image for vulnerabilities.

Install Trivy:
```bash
# macOS
brew install trivy

# Linux
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy
```

### CI/CD Integration

In GitHub Actions or similar CI/CD:

```yaml
- name: Build and push base image
  env:
    DOCKERHUB_USERNAME: ${{ secrets.DOCKERHUB_USERNAME }}
    DOCKERHUB_TOKEN: ${{ secrets.DOCKERHUB_TOKEN }}
  run: |
    echo $DOCKERHUB_TOKEN | docker login -u $DOCKERHUB_USERNAME --password-stdin
    ./scripts/build_and_push_base.sh ${{ github.sha }} $DOCKERHUB_USERNAME
```

### Troubleshooting

**Not logged in to Docker Hub:**
```bash
docker login
```

**Docker not running:**
```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

**Permission denied:**
```bash
chmod +x scripts/build_and_push_base.sh
```

**Build fails:**
- Check Docker disk space: `docker system df`
- Clean up unused images: `docker system prune -a`
- Check logs for specific errors

### Version Strategy

Recommended versioning:
- `latest`: Always points to most recent stable build
- `1.0.0`, `1.1.0`, etc.: Semantic versioning for releases
- `dev`: Development/testing builds
- `SHA`: Commit SHA for exact reproducibility

Example:
```bash
# Production release
./scripts/build_and_push_base.sh 1.0.0 myusername

# Development build
./scripts/build_and_push_base.sh dev myusername

# CI/CD with commit SHA
./scripts/build_and_push_base.sh $(git rev-parse --short HEAD) myusername
```
