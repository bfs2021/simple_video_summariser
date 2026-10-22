#!/usr/bin/env bash
set -u

: "${MIN_FREE_GB:=40}"
: "${MIN_MEM_GB:=12}"
: "${MIN_DOCKER_VERSION:=25.0}"
: "${MIN_COMPOSE_VERSION:=2.24.0}"
ENV_FILE="${ENV_FILE:-.env}"
PREFLIGHT_MEMINFO="${PREFLIGHT_MEMINFO:-/proc/meminfo}"

had_fail=0
net="ytdigest-preflight-net-$$"
cname="ytdigest-preflight-$$"
cleaned=0

pass() { printf 'PASS %s: %s\n' "$1" "$2"; }
warn() { printf 'WARN %s: %s\n' "$1" "$2"; }
fail() { printf 'FAIL %s: %s\n' "$1" "$2"; had_fail=1; }

cleanup() {
	[ "$cleaned" = 1 ] && return
	cleaned=1
	docker rm -f "$cname" >/dev/null 2>&1 || true
	docker network rm "$net" >/dev/null 2>&1 || true
}

interrupted() {
	local code="$1"
	cleanup
	exit "$code"
}

trap cleanup EXIT
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

usage() {
	cat <<'USAGE'
ops/host/preflight.sh [--help]

Checks:
  docker-engine          Docker daemon and server version
  docker-compose         Docker Compose v2 version
  docker-boot            Docker enabled at boot
  disk                   Docker data-root free space
  cpu                    Host CPU count versus transcriber quota
  memory                 Host total memory
  time-sync              NTP synchronization
  env-file               Production environment file
  compose-config         Production Compose configuration
  ollama-from-container  Ollama reachability and model from a container
  ollama-bind            Ollama listening address
  egress                 Required HTTPS destinations from the same container
  docker-group           Root-equivalent Docker group members

Threshold and path variables:
  MIN_FREE_GB=40
  MIN_MEM_GB=12
  MIN_DOCKER_VERSION=25.0
  MIN_COMPOSE_VERSION=2.24.0
  PREFLIGHT_MEMINFO=/proc/meminfo
  ENV_FILE=.env

Values read from .env:
  TRANSCRIBER_CPUS=3.0
  OLLAMA_MODEL=qwen3.5:4b
  SUMMARIZER=ollama
  COMPOSE_FILE
  POSTGRES_PASSWORD
USAGE
}

case "${1:-}" in
	--help | -h)
		trap - EXIT INT TERM
		usage
		exit 0
		;;
	"") ;;
	*)
		trap - EXIT INT TERM
		printf 'error: unknown argument: %s\n\n' "$1" >&2
		usage >&2
		exit 2
		;;
esac

ver_ge() {
	local a="${1#v}" b="${2#v}" i x y
	local IFS=.
	local aa=($a) bb=($b)
	for ((i = 0; i < ${#aa[@]} || i < ${#bb[@]}; i++)); do
		x="${aa[i]:-0}"
		y="${bb[i]:-0}"
		x="${x%%[!0-9]*}"
		y="${y%%[!0-9]*}"
		x="${x:-0}"
		y="${y:-0}"
		if ((10#$x > 10#$y)); then return 0; fi
		if ((10#$x < 10#$y)); then return 1; fi
	done
	return 0
}

env_get() {
	local key="$1" line val
	[ -f "$ENV_FILE" ] || return 0
	line="$(grep -E "^[[:space:]]*${key}=" "$ENV_FILE" | tail -n1)" || true
	val="${line#*=}"
	val="${val%$'\r'}"
	if [[ $val == \"*\" || $val == \'*\' ]]; then val="${val:1:${#val}-2}"; fi
	printf '%s' "$val"
}

check_docker_engine() {
	local version
	if ! version="$(docker version --format '{{.Server.Version}}' 2>/dev/null)" || [ -z "$version" ]; then
		fail docker-engine "Docker daemon not reachable"
	elif ver_ge "$version" "$MIN_DOCKER_VERSION"; then
		pass docker-engine "server $version (>= $MIN_DOCKER_VERSION)"
	else
		fail docker-engine "server $version is below $MIN_DOCKER_VERSION"
	fi
}

check_docker_compose() {
	local version
	if ! version="$(docker compose version --short 2>/dev/null)" || [ -z "$version" ]; then
		fail docker-compose "Docker Compose v2 plugin not found"
	elif ver_ge "$version" "$MIN_COMPOSE_VERSION"; then
		pass docker-compose "$version (>= $MIN_COMPOSE_VERSION)"
	else
		fail docker-compose "$version is below $MIN_COMPOSE_VERSION"
	fi
}

check_docker_boot() {
	local state
	state="$(systemctl is-enabled docker 2>/dev/null)" || true
	if [ "$state" = enabled ]; then
		pass docker-boot "docker.service is enabled"
	else
		fail docker-boot "docker.service is '${state:-unknown}', not enabled"
	fi
}

check_disk() {
	local root free_kb free_gb
	root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" || root=""
	if [ -z "$root" ]; then
		fail disk "cannot read Docker data root"
		return
	fi
	free_kb="$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $4}')"
	if [ -z "$free_kb" ]; then
		fail disk "cannot read free space on $root"
		return
	fi
	free_gb=$((free_kb / 1024 / 1024))
	if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
		pass disk "${free_gb} GB free on $root (>= ${MIN_FREE_GB} GB)"
	else
		fail disk "${free_gb} GB free on $root is below ${MIN_FREE_GB} GB"
	fi
}

check_cpu() {
	local cores cpus
	cores="$(nproc 2>/dev/null)" || cores=0
	cpus="$(env_get TRANSCRIBER_CPUS)"
	cpus="${cpus:-3.0}"
	if awk -v cores="$cores" -v cpus="$cpus" 'BEGIN { exit !(cores + 0 >= cpus + 0) }'; then
		pass cpu "nproc=$cores >= TRANSCRIBER_CPUS=$cpus"
	else
		fail cpu "nproc=$cores < TRANSCRIBER_CPUS=$cpus"
	fi
}

check_memory() {
	local kb gb
	kb="$(awk '/^MemTotal:/ {print $2}' "$PREFLIGHT_MEMINFO" 2>/dev/null)"
	if [ -z "$kb" ]; then
		warn memory "cannot read $PREFLIGHT_MEMINFO"
		return
	fi
	gb=$((kb / 1024 / 1024))
	if [ "$gb" -ge "$MIN_MEM_GB" ]; then
		pass memory "${gb} GB total (>= ${MIN_MEM_GB} GB)"
	else
		warn memory "${gb} GB total is below ${MIN_MEM_GB} GB"
	fi
}

check_time_sync() {
	local state
	state="$(timedatectl show -p NTPSynchronized --value 2>/dev/null)" || true
	if [ "$state" = yes ]; then
		pass time-sync "clock is NTP-synchronised"
	else
		warn time-sync "clock not NTP-synchronised"
	fi
}

check_env_file() {
	if [ ! -f "$ENV_FILE" ]; then
		fail env-file "$ENV_FILE not found"
		return
	fi
	local mode password compose_file ok=1
	mode="$(stat -c '%a' "$ENV_FILE" 2>/dev/null)"
	if [ "$mode" != 600 ]; then
		fail env-file "$ENV_FILE mode is ${mode:-unknown}, must be 600"
		ok=0
	fi
	compose_file="$(env_get COMPOSE_FILE)"
	if [ "$compose_file" != compose.yml ]; then
		fail env-file "COMPOSE_FILE must be compose.yml"
		ok=0
	fi
	password="$(env_get POSTGRES_PASSWORD)"
	if [ -z "$password" ]; then
		fail env-file "POSTGRES_PASSWORD is empty"
		ok=0
	elif ! [[ $password =~ ^[A-Za-z0-9_-]+$ ]]; then
		fail env-file "POSTGRES_PASSWORD contains invalid characters"
		ok=0
	fi
	[ "$ok" = 1 ] && pass env-file "mode 600 and production settings valid"
}

check_compose_config() {
	local output
	if ! output="$(docker compose config 2>&1)"; then
		fail compose-config "docker compose config failed"
	elif grep -q 'type: bind' <<<"$output"; then
		fail compose-config "config contains a bind mount"
	elif grep -qE 'WHISPER_MODEL:[[:space:]]*"?tiny"?' <<<"$output"; then
		fail compose-config "config has WHISPER_MODEL: tiny"
	else
		pass compose-config "no bind mount or tiny Whisper model"
	fi
}

check_ollama_bind() {
	local listeners
	listeners="$(ss -ltnH 2>/dev/null | awk '$4 ~ /:11434$/ {print $4}')"
	if [ -z "$listeners" ]; then
		fail ollama-bind "nothing listens on port 11434"
	elif grep -Eq '^(0\.0\.0\.0|\[::\]|\*):11434$' <<<"$listeners"; then
		warn ollama-bind "wildcard listener; verify the runbook firewall rules"
	else
		pass ollama-bind "listener is not wildcard: $(tr '\n' ',' <<<"$listeners" | sed 's/,$//')"
	fi
}

container_output=""
container_ready=0
run_container_checks() {
	local model summarizer hosts py
	model="$(env_get OLLAMA_MODEL)"
	model="${model:-qwen3.5:4b}"
	summarizer="$(env_get SUMMARIZER)"
	hosts="www.youtube.com,rr1---sn.example.googlevideo.com,huggingface.co,cdn-lfs.huggingface.co,pypi.org,files.pythonhosted.org,deb.debian.org,registry.npmjs.org,registry-1.docker.io"
	[ "$summarizer" = anthropic ] && hosts="$hosts,api.anthropic.com"
	if ! docker network create "$net" >/dev/null 2>&1; then
		fail ollama-from-container "could not create throwaway network"
		fail egress "could not create throwaway network"
		return
	fi
	py='import json,os,urllib.error,urllib.request
model=os.environ["PREFLIGHT_OLLAMA_MODEL"]
hosts=os.environ["PREFLIGHT_EGRESS_HOSTS"].split(",")
try:
 with urllib.request.urlopen("http://host.docker.internal:11434/api/tags",timeout=10) as r:
  data=json.load(r); print("OLLAMA\t%s\t%s"%(r.status,",".join(x.get("model","") for x in data.get("models",[]))))
except Exception:
 print("OLLAMA\tERR\t")
for host in hosts:
 try:
  with urllib.request.urlopen("https://"+host,timeout=10) as r: r.read(1)
  print("EGRESS\t%s\tOK"%host)
 except urllib.error.HTTPError:
  print("EGRESS\t%s\tOK"%host)
 except Exception:
  print("EGRESS\t%s\tFAIL"%host)'
	container_output="$(docker run --pull=never --rm --name "$cname" --network "$net" --add-host host.docker.internal:host-gateway -e "PREFLIGHT_OLLAMA_MODEL=$model" -e "PREFLIGHT_EGRESS_HOSTS=$hosts" python:3.14-slim python -c "$py" 2>/dev/null)" || true
	container_ready=1
}

check_ollama_from_container() {
	local model record status models
	model="$(env_get OLLAMA_MODEL)"
	model="${model:-qwen3.5:4b}"
	[ "$container_ready" = 1 ] || return
	record="$(awk -F '\t' '$1 == "OLLAMA" {print; exit}' <<<"$container_output")"
	status="$(cut -f2 <<<"$record")"
	models="$(cut -f3- <<<"$record")"
	if [ "$status" != 200 ]; then
		fail ollama-from-container "container cannot reach host Ollama"
	elif grep -Fxq -- "$model" <<<"${models//,/$'\n'}"; then
		pass ollama-from-container "reachable and model '$model' is present"
	else
		warn ollama-from-container "reachable but model '$model' is missing"
	fi
}

check_egress() {
	local summarizer hosts host failures=""
	summarizer="$(env_get SUMMARIZER)"
	hosts="www.youtube.com rr1---sn.example.googlevideo.com huggingface.co cdn-lfs.huggingface.co pypi.org files.pythonhosted.org deb.debian.org registry.npmjs.org registry-1.docker.io"
	[ "$summarizer" = anthropic ] && hosts="$hosts api.anthropic.com"
	[ "$container_ready" = 1 ] || return
	for host in $hosts; do
		if ! awk -F '\t' -v host="$host" '$1 == "EGRESS" && $2 == host && $3 == "OK" {found=1} END {exit !found}' <<<"$container_output"; then
			failures="${failures}${failures:+, }$host"
		fi
	done
	if [ -n "$failures" ]; then
		fail egress "unreachable: $failures"
	else
		pass egress "reachable: ${hosts// /, }"
	fi
}

check_docker_group() {
	local group members
	group="$(getent group docker 2>/dev/null)" || true
	members="${group##*:}"
	pass docker-group "members: ${members:-none} (root-equivalent access)"
}

check_docker_engine
check_docker_compose
check_docker_boot
check_disk
check_cpu
check_memory
check_time_sync
check_env_file
check_compose_config
check_ollama_bind
run_container_checks
check_ollama_from_container
check_egress
check_docker_group

exit "$had_fail"
