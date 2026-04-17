{{/*
Expand the name of the chart.
*/}}
{{- define "ai-apis.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec). If the release name already contains the chart name
it will be used as-is.
*/}}
{{- define "ai-apis.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label value — used in the helm.sh/chart annotation.
*/}}
{{- define "ai-apis.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "ai-apis.labels" -}}
helm.sh/chart: {{ include "ai-apis.chart" . }}
{{ include "ai-apis.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels used in matchLabels / Service selectors.
Both an app.kubernetes.io/name and app.kubernetes.io/instance label are
included so that multiple releases of the chart can coexist in the same
namespace without colliding.
*/}}
{{- define "ai-apis.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ai-apis.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Render a full image reference:  registry/repository/imageName:tag
Usage:
  {{ include "ai-apis.image" (dict "root" . "imageName" .Values.whisper.image) }}
*/}}
{{- define "ai-apis.image" -}}
{{- $registry   := .root.Values.global.imageRegistry   | default "docker.io" -}}
{{- $repository := .root.Values.global.imageRepository | required "global.imageRepository must be set" -}}
{{- $tag        := .root.Values.global.imageTag        | default "latest" -}}
{{- printf "%s/%s/%s:%s" $registry $repository .imageName $tag -}}
{{- end }}

{{/*
Node selector that schedules a pod onto a node with an NVIDIA GPU.
*/}}
{{- define "ai-apis.gpuNodeSelector" -}}
nvidia.com/gpu.present: "true"
{{- end }}

{{/*
Construct the MongoDB connection URL.
If .Values.global.mongodbUrl is non-empty that value is returned unchanged.
Otherwise the URL is built from the mongodb sub-values and the release name.

The URL intentionally omits the database name so that the application can
append it, which is the convention used by the docker-compose configuration.
*/}}
{{/*
Keycloak environment variables block.
Renders nothing when keycloak.url is empty — all existing API key auth is preserved.
Include in a container's env list with:
  {{- include "ai-apis.keycloakEnv" . | nindent 12 }}
*/}}
{{- define "ai-apis.keycloakEnv" -}}
{{- if .Values.keycloak.url }}
- name: KEYCLOAK_URL
  value: {{ .Values.keycloak.url | quote }}
- name: KEYCLOAK_REALM
  value: {{ .Values.keycloak.realm | quote }}
- name: KEYCLOAK_CLIENT_ID
  value: {{ .Values.keycloak.clientId | quote }}
- name: KEYCLOAK_CLIENT_SECRET
  valueFrom:
    secretKeyRef:
      name: ai-apis-secrets
      key: keycloakClientSecret
- name: KEYCLOAK_ADMIN_ROLE
  value: {{ .Values.keycloak.adminRole | quote }}
- name: KEYCLOAK_VERIFY_SSL
  value: {{ .Values.keycloak.verifySSL | quote }}
{{- end }}
{{- end }}

{{- define "ai-apis.mongodbUrl" -}}
{{- if .Values.global.mongodbUrl -}}
{{- .Values.global.mongodbUrl -}}
{{- else -}}
{{- printf "mongodb://$(MONGO_USER):$(MONGO_PASSWORD)@%s-mongodb:27017/" .Release.Name -}}
{{- end -}}
{{- end }}
