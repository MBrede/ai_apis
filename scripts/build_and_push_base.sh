#!/bin/bash
#
# Build and push base Docker image to Docker Hub
#
# This base image contains ONLY system-level dependencies:
#   - Python 3.12
#   - CUDA 13.0.2 + cuDNN
#   - uv (Python package installer)
#   - System packages (ffmpeg, git, curl, etc.)
#
# Python packages are installed in each service-specific Dockerfile.
# This approach allows:
#   - ONE base image for ALL services (no duplication)
#   - Each service installs only the Python packages it needs
#   - Smaller base image that changes infrequently
#
# Usage:
#   ./scripts/build_and_push_base.sh [VERSION] [DOCKERHUB_USERNAME]
#
# Example:
#   ./scripts/build_and_push_base.sh 1.0.0 myusername
#

set -e  # Exit on error

# Configuration
VERSION=${1:-"latest"}
DOCKERHUB_USERNAME=${2:-${DOCKERHUB_USERNAME:-""}}
IMAGE_NAME="ai-apis-base"
FULL_IMAGE_NAME="${DOCKERHUB_USERNAME}/${IMAGE_NAME}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate Docker Hub username
if [ -z "$DOCKERHUB_USERNAME" ]; then
    log_error "Docker Hub username not provided"
    echo "Usage: $0 [VERSION] [DOCKERHUB_USERNAME]"
    echo "Or set DOCKERHUB_USERNAME environment variable"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if logged in to Docker Hub
if ! docker info | grep -q "Username"; then
    log_warning "Not logged in to Docker Hub. Attempting to login..."
    docker login
fi

log_info "Building base image: ${FULL_IMAGE_NAME}:${VERSION}"
log_info "This may take several minutes..."

# Build the base image
cd "$(dirname "$0")/.."
docker build \
    -f docker/Dockerfile.base \
    -t "${FULL_IMAGE_NAME}:${VERSION}" \
    -t "${FULL_IMAGE_NAME}:latest" \
    --progress=plain \
    .

log_success "Base image built successfully"

# Show image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}:${VERSION}" --format "{{.Size}}")
log_info "Image size: ${IMAGE_SIZE}"

# Push to Docker Hub
log_info "Pushing ${FULL_IMAGE_NAME}:${VERSION} to Docker Hub..."
docker push "${FULL_IMAGE_NAME}:${VERSION}"

if [ "$VERSION" != "latest" ]; then
    log_info "Pushing ${FULL_IMAGE_NAME}:latest to Docker Hub..."
    docker push "${FULL_IMAGE_NAME}:latest"
fi

log_success "Successfully pushed base image to Docker Hub"
log_info "Image: ${FULL_IMAGE_NAME}:${VERSION}"

# Display usage instructions
echo ""
log_info "To use this base image in your Dockerfiles, update the FROM statement:"
echo ""
echo "  FROM ${FULL_IMAGE_NAME}:${VERSION}"
echo ""
log_info "Or use 'latest' for the most recent version:"
echo ""
echo "  FROM ${FULL_IMAGE_NAME}:latest"
echo ""

# Optional: Scan for vulnerabilities (requires Docker Scout or Trivy)
if command -v trivy &> /dev/null; then
    log_info "Scanning image for vulnerabilities with Trivy..."
    trivy image "${FULL_IMAGE_NAME}:${VERSION}"
else
    log_warning "Trivy not installed. Skipping vulnerability scan."
    log_info "Install Trivy for security scanning: https://github.com/aquasecurity/trivy"
fi

log_success "Build and push completed successfully!"
