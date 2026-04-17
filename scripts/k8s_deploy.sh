#!/usr/bin/env bash
# k8s_deploy.sh — Build, push, and generate Helm values from .env
#
# Usage:
#   ./scripts/k8s_deploy.sh [OPTIONS]
#
# Options:
#   --values-only    Generate my-values.yaml only, skip build and push
#   --no-push        Build images but do not push to registry
#   --tag TAG        Override image tag (default: git SHA or "latest")
#   --env FILE       Path to .env file (default: .env)
#   -h, --help       Show this help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VALUES_ONLY=false
NO_PUSH=false
ENV_FILE=".env"
TAG=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_VALUES="${ROOT_DIR}/my-values.yaml"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --values-only) VALUES_ONLY=true; shift ;;
        --no-push)     NO_PUSH=true;     shift ;;
        --tag)         TAG="$2";         shift 2 ;;
        --env)         ENV_FILE="$2";    shift 2 ;;
        -h|--help)
            head -14 "${BASH_SOURCE[0]}" | tail -12
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
ENV_FILE="${ROOT_DIR}/${ENV_FILE#./}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found. Copy .env.example to .env and fill in values." >&2
    exit 1
fi

# Export all variables from .env (handles key=value and key="value")
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# ---------------------------------------------------------------------------
# Resolve image tag
# ---------------------------------------------------------------------------
if [[ -z "${TAG}" ]]; then
    if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree &>/dev/null; then
        TAG="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
    else
        TAG="latest"
    fi
fi

# ---------------------------------------------------------------------------
# Validate required variables
# ---------------------------------------------------------------------------
ERRORS=()

[[ -z "${DOCKER_HUB_USER:-}" ]] && ERRORS+=("DOCKER_HUB_USER is not set in ${ENV_FILE}")
[[ -z "${API_KEY:-}"         ]] && ERRORS+=("API_KEY is not set")
[[ -z "${TELEGRAM_TOKEN:-}"  ]] && ERRORS+=("TELEGRAM_TOKEN is not set")
[[ -z "${HF_TOKEN:-}"        ]] && ERRORS+=("HF_TOKEN is not set (required for Whisper diarization model)")

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "ERROR: Missing required configuration:" >&2
    for e in "${ERRORS[@]}"; do echo "  - ${e}" >&2; done
    exit 1
fi

REGISTRY="docker.io"
REPO="${DOCKER_HUB_USER}"

# ---------------------------------------------------------------------------
# Image definitions: name → dockerfile
# ---------------------------------------------------------------------------
declare -A IMAGES=(
    [ai-apis-whisper]="docker/Dockerfile.whisper.hub"
    [ai-apis-stable-diffusion]="docker/Dockerfile.stable_diffusion.hub"
    [ai-apis-text-classification]="docker/Dockerfile.text_analysis.hub"
    [ai-apis-bot]="docker/Dockerfile.bot"
    [ai-apis-nextcloud]="docker/Dockerfile.nextcloud"
)

# Preserve insertion order for readable output
IMAGE_ORDER=(
    ai-apis-whisper
    ai-apis-stable-diffusion
    ai-apis-text-classification
    ai-apis-bot
    ai-apis-nextcloud
)

# ---------------------------------------------------------------------------
# Build images
# ---------------------------------------------------------------------------
if [[ "${VALUES_ONLY}" == "false" ]]; then
    echo "==> Building images (tag: ${TAG})"
    for name in "${IMAGE_ORDER[@]}"; do
        dockerfile="${IMAGES[$name]}"
        full_image="${REGISTRY}/${REPO}/${name}:${TAG}"
        echo ""
        echo "--- Building ${full_image} from ${dockerfile}"
        docker build \
            --file "${ROOT_DIR}/${dockerfile}" \
            --tag  "${full_image}" \
            "${ROOT_DIR}"
    done
fi

# ---------------------------------------------------------------------------
# Push images
# ---------------------------------------------------------------------------
if [[ "${VALUES_ONLY}" == "false" && "${NO_PUSH}" == "false" ]]; then
    echo ""
    echo "==> Pushing images to ${REGISTRY}/${REPO}"
    for name in "${IMAGE_ORDER[@]}"; do
        full_image="${REGISTRY}/${REPO}/${name}:${TAG}"
        echo "--- Pushing ${full_image}"
        docker push "${full_image}"
    done
fi

# ---------------------------------------------------------------------------
# Generate my-values.yaml
# ---------------------------------------------------------------------------
echo ""
echo "==> Writing ${OUTPUT_VALUES}"

# Helper: emit a YAML string value, or empty quotes if unset
val() { printf '%s' "${1:-}"; }

cat > "${OUTPUT_VALUES}" << YAML
# my-values.yaml — generated by scripts/k8s_deploy.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Do NOT commit this file — it contains secrets.

global:
  imageRegistry: "${REGISTRY}"
  imageRepository: "$(val "${DOCKER_HUB_USER}")"
  imageTag: "${TAG}"
  storageClass: longhorn
  apiKey: "$(val "${API_KEY}")"
  adminApiKey: "$(val "${ADMIN_API_KEY:-}")"
  useMongodb: ${USE_MONGODB:-true}
  mongodbUrl: ""   # auto-constructed from mongodb.rootUser / rootPassword below

keycloak:
  url: "$(val "${KEYCLOAK_URL:-}")"
  realm: "$(val "${KEYCLOAK_REALM:-master}")"
  clientId: "$(val "${KEYCLOAK_CLIENT_ID:-}")"
  clientSecret: "$(val "${KEYCLOAK_CLIENT_SECRET:-}")"
  adminRole: "$(val "${KEYCLOAK_ADMIN_ROLE:-ai-apis-admin}")"
  verifySSL: ${KEYCLOAK_VERIFY_SSL:-false}

mongodb:
  rootUser: "$(val "${MONGO_ROOT_USER:-admin}")"
  rootPassword: "$(val "${MONGO_ROOT_PASSWORD:-changeme}")"
  database: "$(val "${MONGODB_DB:-ai_apis}")"
  storage: 10Gi

whisper:
  image: ai-apis-whisper
  port: ${WHISPER_PORT:-8080}
  hfToken: "$(val "${HF_TOKEN}")"
  cacheStorage: 20Gi
  defaultModel: "$(val "${DEFAULT_WHISPER_MODEL:-turbo}")"
  ingress:
    host: "$(val "${WHISPER_INGRESS_HOST:-whisper.example.com}")"
    tls: ${WHISPER_INGRESS_TLS:-false}

stableDiffusion:
  image: ai-apis-stable-diffusion
  port: ${SD_PORT:-1234}
  hfToken: "$(val "${HF_TOKEN}")"
  civitKey: "$(val "${CIVIT_KEY:-}")"
  cacheStorage: 20Gi
  modelsStorage: 50Gi
  lorasStorage: 5Gi
  ingress:
    host: "$(val "${SD_INGRESS_HOST:-sd.example.com}")"
    tls: ${SD_INGRESS_TLS:-false}

textClassification:
  image: ai-apis-text-classification
  port: 8000
  cacheStorage: 10Gi

telegramBot:
  image: ai-apis-bot
  token: "$(val "${TELEGRAM_TOKEN}")"
  ollamaHost: "$(val "${OLLAMA_HOST:-}")"
  ollamaPort: ${OLLAMA_PORT:-2345}

nextcloudSync:
  image: ai-apis-nextcloud
  schedule: "$(val "${NEXTCLOUD_SCHEDULE:-0 2 * * *}")"
  nextcloudUrl: "$(val "${NEXTCLOUD_URL:-}")"
  nextcloudUser: "$(val "${NEXTCLOUD_USER:-}")"
  nextcloudPassword: "$(val "${NEXTCLOUD_PASSWORD:-}")"
  nextcloudFolder: "$(val "${NEXTCLOUD_FOLDER:-}")"
  numSpeakers: "$(val "${NUM_SPEAKERS:-}")"
  minSpeakers: "$(val "${MIN_SPEAKERS:-}")"
  maxSpeakers: "$(val "${MAX_SPEAKERS:-}")"
  whisperTimeout: "$(val "${WHISPER_TIMEOUT:-3600}")"
YAML

echo "Done. To deploy:"
echo ""
echo "  helm upgrade --install ai-apis helm/ai-apis -f my-values.yaml"
echo ""
echo "To update after code changes:"
echo ""
echo "  ./scripts/k8s_deploy.sh && helm upgrade ai-apis helm/ai-apis -f my-values.yaml"
