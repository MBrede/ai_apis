#!/usr/bin/env bash
# k8s_deploy.sh — Build, push, and generate Helm values from .env
#
# Usage:
#   ./scripts/k8s_deploy.sh [OPTIONS]
#
# Options:
#   --values-only          Generate my-values.yaml only, skip build and push
#   --no-push              Build images but do not push to registry
#   --tag TAG              Override image tag (default: git SHA or "latest")
#   --env FILE             Path to .env file (default: .env)
#   --ignore-env VAR,...   Comma-separated extra .env vars to ignore when
#                          generating my-values.yaml (in addition to the
#                          built-in ignore list below)
#   -h, --help             Show this help
#
# Built-in ignored vars (docker-compose specific, wrong for k8s):
#   MONGODB_URL   — localhost URL; helm constructs the in-cluster URL instead
#   WHISPER_HOST  — compose service name; k8s uses the Service DNS name
#   WHISPER_PORT  — only the container port matters; ingress handles the rest
#   SD_HOST       — same as WHISPER_HOST
#   SD_PORT       — same as WHISPER_PORT

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

# Vars that are meaningful for docker-compose but wrong/misleading in k8s.
# These are unset after sourcing .env so they never leak into my-values.yaml.
BUILTIN_IGNORE=(
    MONGODB_URL
    WHISPER_HOST
    SD_HOST
)

EXTRA_IGNORE=()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --values-only) VALUES_ONLY=true; shift ;;
        --no-push)     NO_PUSH=true;     shift ;;
        --tag)         TAG="$2";         shift 2 ;;
        --env)         ENV_FILE="$2";    shift 2 ;;
        --ignore-env)
            IFS=',' read -ra EXTRA_IGNORE <<< "$2"
            shift 2
            ;;
        -h|--help)
            head -22 "${BASH_SOURCE[0]}" | tail -20
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

# Export all variables from .env.
# Parses KEY=VALUE manually instead of sourcing the file directly so that
# unquoted values containing spaces (e.g. cron schedules "0 2 * * *") are not
# split by bash and executed as commands.
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    # Must match KEY=VALUE
    [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*) ]] || continue
    key="${BASH_REMATCH[1]}"
    val="${BASH_REMATCH[2]}"
    # Strip surrounding single or double quotes if present
    if [[ "$val" =~ ^\"(.*)\"$ ]] || [[ "$val" =~ ^\'(.*)\'$ ]]; then
        val="${BASH_REMATCH[1]}"
    fi
    export "$key=$val"
done < "${ENV_FILE}"

# Unset vars that are docker-compose-specific / wrong for k8s
ALL_IGNORE=("${BUILTIN_IGNORE[@]}" "${EXTRA_IGNORE[@]}")
if [[ ${#ALL_IGNORE[@]} -gt 0 ]]; then
    echo "==> Ignoring .env vars: ${ALL_IGNORE[*]}"
    for var in "${ALL_IGNORE[@]}"; do
        unset "${var}" 2>/dev/null || true
    done
fi

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
[[ -z "${TELEGRAM_TOKEN:-}"  ]] && ERRORS+=("TELEGRAM_TOKEN is not set")
[[ -z "${HF_TOKEN:-}"        ]] && ERRORS+=("HF_TOKEN is not set (required for Whisper diarization model)")

# API_KEY only required when Keycloak is not handling auth
if [[ -z "${KEYCLOAK_URL:-}" && -z "${API_KEY:-}" ]]; then
    ERRORS+=("API_KEY is not set and KEYCLOAK_URL is not configured — one of them is required")
fi

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

# Helper: emit a YAML string value, or empty string if unset
val() { printf '%s' "${1:-}"; }

# When Keycloak is configured, hardcoded API keys are not needed — access is
# managed per-consumer via Keycloak clients. Emit empty strings so the Helm
# secret still renders but the values carry no sensitive data.
if [[ -n "${KEYCLOAK_URL:-}" ]]; then
    EFFECTIVE_API_KEY=""
    EFFECTIVE_ADMIN_KEY=""
    echo "==> Keycloak configured — omitting API_KEY and ADMIN_API_KEY from values"
else
    EFFECTIVE_API_KEY="${API_KEY:-}"
    EFFECTIVE_ADMIN_KEY="${ADMIN_API_KEY:-}"
fi

cat > "${OUTPUT_VALUES}" << YAML
# my-values.yaml — generated by scripts/k8s_deploy.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Do NOT commit this file — it contains secrets.

global:
  imageRegistry: "${REGISTRY}"
  imageRepository: "$(val "${DOCKER_HUB_USER}")"
  imageTag: "${TAG}"
  storageClass: longhorn
  apiKey: "$(val "${EFFECTIVE_API_KEY}")"
  adminApiKey: "$(val "${EFFECTIVE_ADMIN_KEY}")"
  useMongodb: ${USE_MONGODB:-true}
  mongodbUrl: ""   # intentionally empty — helm constructs the in-cluster URL

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
  port: 8080
  hfToken: "$(val "${HF_TOKEN}")"
  cacheStorage: 20Gi
  defaultModel: "$(val "${DEFAULT_WHISPER_MODEL:-turbo}")"
  ingress:
    host: "$(val "${WHISPER_INGRESS_HOST:-whisper.example.com}")"
    tls: ${WHISPER_INGRESS_TLS:-false}
    internalHost: "$(val "${WHISPER_INTERNAL_HOST:-whisper-internal.cluster.local}")"

stableDiffusion:
  image: ai-apis-stable-diffusion
  port: 1234
  hfToken: "$(val "${HF_TOKEN}")"
  civitKey: "$(val "${CIVIT_KEY:-}")"
  cacheStorage: 20Gi
  modelsStorage: 50Gi
  lorasStorage: 5Gi
  ingress:
    host: "$(val "${SD_INGRESS_HOST:-sd.example.com}")"
    tls: ${SD_INGRESS_TLS:-false}
    internalHost: "$(val "${SD_INTERNAL_HOST:-sd-internal.cluster.local}")"

textClassification:
  image: ai-apis-text-classification
  port: 8000
  cacheStorage: 10Gi
  ingress:
    host: "$(val "${TEXTCLASS_INGRESS_HOST:-textclass.cluster.local}")"

keda:
  enabled: ${KEDA_ENABLED:-false}
  idleTimeout: ${KEDA_IDLE_TIMEOUT:-1800}
  scalingTimeout: ${KEDA_SCALING_TIMEOUT:-300}

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
