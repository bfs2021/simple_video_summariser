#!/usr/bin/env bash
#
# Deploy-host preflight for ytdigest (task #78). Run from the checkout root:
#
#     ops/host/preflight.sh [--help]
#
# It reports PASS | WARN | FAIL for each host requirement from
# _docs/deploy-host.md, so the owner can tell whether the host is ready for
# the #3 bake-off and the #58 stack before the first deploy.
#
# It is READ-ONLY on the host: it installs nothing, edits no config and never
# runs sudo. The only things it creates are one uniquely named throwaway Docker
# network and one throwaway container (for the ollama-from-container check); a
# trap removes both on exit, including on interrupt. It never prints a secret.
#
# Exit status: 0 when no check FAILs (WARNs allowed), 1 when any check FAILs,
# 2 on a usage error (an unknown flag).

set -u

# --- thresholds (override by exporting the variable before running) ----------
: "${MIN_FREE_GB:=40}"     # disk: free GB on the Docker data root filesystem
: "${MIN_MEM_GB:=12}"      # memory: total RAM floor (WARN below)
: "${DOCKER_ENGINE_MIN:=25.0}"    # docker-engine: server version floor
: "${DOCKER_COMPOSE_MIN:=2.24.0}" # docker-compose: plugin version floor
ENV_FILE="${ENV_FILE:-.env}"

# --- output helpers ----------------------------------------------------------
had_fail=0
pass() { printf 'PASS %s: %s\n' "$1" "$2"; }
warn() { printf 'WARN %s: %s\n' "$1" "$2"; }
fail() { printf 'FAIL %s: %s\n' "$1" "$2"; had_fail=1; }

usage() {
	cat <<'USAGE'
ops/host/preflight.sh [--help]

Reports PASS | WARN | FAIL for each deploy-host requirement. Read-only: it
installs nothing, edits no config and never runs sudo. Exit 0 if no check
FAILs, 1 if any FAILs, 2 on a usage error.

Checks:
  docker-engine        Docker daemon reachable, server version >= DOCKER_ENGINE_MIN
  docker-compose       `docker compose version` >= DOCKER_COMPOSE_MIN (numeric compare)
  docker-boot          `systemctl is-enabled docker` is `enabled`
  disk                 free space on the Docker data root >= MIN_FREE_GB (FAIL below)
  cpu                  nproc >= TRANSCRIBER_CPUS from .env (Docker rejects a quota above nproc)
  memory               total RAM >= MIN_MEM_GB (WARN below, not FAIL)
  time-sync            clock is NTP-synchronised (WARN otherwise)
  env-file             .env exists, mode 600, sets COMPOSE_FILE=compose.yml, valid POSTGRES_PASSWORD
  compose-config       `docker compose config` ok, no bind mount, no WHISPER_MODEL: tiny
  ollama-from-container  a throwaway container reaches host Ollama via host.docker.internal:11434

Threshold variables (current defaults):
  MIN_FREE_GB=40  MIN_MEM_GB=12  DOCKER_ENGINE_MIN=25.0  DOCKER_COMPOSE_MIN=2.24.0
  ENV_FILE=.env
Values read from .env: TRANSCRIBER_CPUS (default 3.0), OLLAMA_MODEL (default qwen3.5:4b),
  COMPOSE_FILE, POSTGRES_PASSWORD.
USAGE
}

case "${1:-}" in
	--help | -h)
		usage
		exit 0
		;;
	"") ;;
	*)
		printf 'error: unknown argument: %s\n\n' "$1" >&2
		usage >&2
		exit 2
		;;
esac

# --- small utilities ---------------------------------------------------------

# ver_ge A B -> 0 (true) if version A >= B, comparing dot-separated numeric
# fields, so 2.100.1 >= 2.24.0 and 2.23.9 < 2.24.0. A leading `v` is stripped.
ver_ge() {
	local a="${1#v}" b="${2#v}"
	local IFS=.
	# shellcheck disable=SC2206
	local aa=($a) bb=($b) i
	for ((i = 0; i < ${#aa[@]} || i < ${#bb[@]}; i++)); do
		local x="${aa[i]:-0}" y="${bb[i]:-0}"
		# strip any non-digit suffix (e.g. a build tag) from each field
		x="${x%%[!0-9]*}"
		y="${y%%[!0-9]*}"
		x="${x:-0}"
		y="${y:-0}"
		if ((10#$x > 10#$y)); then return 0; fi
		if ((10#$x < 10#$y)); then return 1; fi
	done
	return 0
}

# env_get KEY -> value of KEY in $ENV_FILE, without executing the file. Takes
# the last assignment, strips surrounding quotes. Empty if absent.
env_get() {
	local key="$1" line val
	[ -f "$ENV_FILE" ] || return 0
	line="$(grep -E "^[[:space:]]*${key}=" "$ENV_FILE" | tail -n1)" || true
	val="${line#*=}"
	val="${val%$'\r'}"
	# strip one layer of matching quotes
	if [[ $val == \"*\" || $val == \'*\' ]]; then val="${val:1:${#val}-2}"; fi
	printf '%s' "$val"
}

# --- checks ------------------------------------------------------------------

check_docker_engine() {
	local v
	if ! v="$(docker version --format '{{.Server.Version}}' 2>/dev/null)" || [ -z "$v" ]; then
		fail docker-engine "Docker daemon not reachable (docker version failed)"
		return
	fi
	if ver_ge "$v" "$DOCKER_ENGINE_MIN"; then
		pass docker-engine "server $v (>= $DOCKER_ENGINE_MIN)"
	else
		fail docker-engine "server $v is below the $DOCKER_ENGINE_MIN floor"
	fi
}

check_docker_compose() {
	local v
	if ! v="$(docker compose version --short 2>/dev/null)" || [ -z "$v" ]; then
		fail docker-compose "docker compose v2 plugin not found (docker compose version failed)"
		return
	fi
	if ver_ge "$v" "$DOCKER_COMPOSE_MIN"; then
		pass docker-compose "$v (>= $DOCKER_COMPOSE_MIN)"
	else
		fail docker-compose "$v is below the $DOCKER_COMPOSE_MIN floor"
	fi
}

check_docker_boot() {
	local s
	s="$(systemctl is-enabled docker 2>/dev/null)" || true
	if [ "$s" = enabled ]; then
		pass docker-boot "docker.service is enabled at boot"
	else
		fail docker-boot "docker.service is '${s:-unknown}', not 'enabled' (restart: unless-stopped won't survive a reboot)"
	fi
}

check_disk() {
	local root free_kb free_gb
	root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" || root=""
	if [ -z "$root" ]; then
		fail disk "cannot read Docker data root (docker info failed)"
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
	# compare as floats via awk; equal passes
	if awk -v c="$cores" -v q="$cpus" 'BEGIN { exit !(c+0 >= q+0) }'; then
		pass cpu "nproc=$cores >= TRANSCRIBER_CPUS=$cpus"
	else
		fail cpu "nproc=$cores < TRANSCRIBER_CPUS=$cpus (Docker rejects a CPU quota above the host count)"
	fi
}

check_memory() {
	local kb gb
	kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)"
	if [ -z "$kb" ]; then
		warn memory "cannot read /proc/meminfo"
		return
	fi
	gb=$((kb / 1024 / 1024))
	if [ "$gb" -ge "$MIN_MEM_GB" ]; then
		pass memory "${gb} GB total (>= ${MIN_MEM_GB} GB)"
	else
		warn memory "${gb} GB total is below ${MIN_MEM_GB} GB (transcriber 6G + Ollama + Postgres + rest)"
	fi
}

check_time_sync() {
	local s
	s="$(timedatectl show -p NTPSynchronized --value 2>/dev/null)" || true
	if [ "$s" = yes ]; then
		pass time-sync "clock is NTP-synchronised"
	else
		warn time-sync "clock not NTP-synchronised (NTPSynchronized=${s:-unknown})"
	fi
}

check_env_file() {
	if [ ! -f "$ENV_FILE" ]; then
		fail env-file "$ENV_FILE not found in the checkout root"
		return
	fi
	local mode pw compose_file ok=1
	mode="$(stat -c '%a' "$ENV_FILE" 2>/dev/null)"
	if [ "$mode" != 600 ]; then
		fail env-file "$ENV_FILE mode is ${mode:-unknown}, must be 600 (group/world must not read it)"
		ok=0
	fi
	compose_file="$(env_get COMPOSE_FILE)"
	if [ "$compose_file" != compose.yml ]; then
		fail env-file "COMPOSE_FILE must be compose.yml (is '${compose_file:-unset}'), so a host deploy never merges compose.override.yml"
		ok=0
	fi
	pw="$(env_get POSTGRES_PASSWORD)"
	if [ -z "$pw" ]; then
		fail env-file "POSTGRES_PASSWORD is empty"
		ok=0
	elif ! [[ $pw =~ ^[A-Za-z0-9_-]+$ ]]; then
		# never print the value
		fail env-file "POSTGRES_PASSWORD contains characters outside [A-Za-z0-9_-] (it is inserted unencoded into DATABASE_URL)"
		ok=0
	fi
	[ "$ok" = 1 ] && pass env-file "mode 600, COMPOSE_FILE=compose.yml, POSTGRES_PASSWORD valid"
}

check_compose_config() {
	local out
	if ! out="$(docker compose config 2>&1)"; then
		fail compose-config "docker compose config failed from the checkout root"
		return
	fi
	if grep -q 'type: bind' <<<"$out"; then
		fail compose-config "config contains a bind mount (the dev overlay compose.override.yml is being merged)"
		return
	fi
	if grep -qE 'WHISPER_MODEL:[[:space:]]*"?tiny"?' <<<"$out"; then
		fail compose-config "config has WHISPER_MODEL: tiny (the dev overlay is being merged)"
		return
	fi
	pass compose-config "no bind mount, no WHISPER_MODEL: tiny (dev overlay not merged)"
}

check_ollama_from_container() {
	local net cname model img='python:3.14-slim'
	net="ytdigest-preflight-net-$$"
	cname="ytdigest-preflight-$$"
	model="$(env_get OLLAMA_MODEL)"
	model="${model:-qwen3.5:4b}"

	# shellcheck disable=SC2317
	cleanup_ollama() {
		docker rm -f "$cname" >/dev/null 2>&1 || true
		docker network rm "$net" >/dev/null 2>&1 || true
	}
	trap cleanup_ollama RETURN

	if ! docker network create "$net" >/dev/null 2>&1; then
		fail ollama-from-container "could not create a throwaway Docker network"
		return
	fi

	# One container, stdlib urllib only: print the HTTP status, then the body's
	# model names. host.docker.internal resolves via --add-host host-gateway.
	local py='import json,sys,urllib.request
try:
    with urllib.request.urlopen("http://host.docker.internal:11434/api/tags", timeout=10) as r:
        print(r.status)
        data=json.load(r)
        print(",".join(m.get("model","") for m in data.get("models",[])))
except Exception as e:
    print("ERR"); print(str(e))'

	local out status tags
	out="$(docker run --rm --name "$cname" --network "$net" \
		--add-host host.docker.internal:host-gateway \
		"$img" python -c "$py" 2>/dev/null)" || true
	status="$(sed -n '1p' <<<"$out")"
	tags="$(sed -n '2p' <<<"$out")"

	if [ "$status" != 200 ]; then
		fail ollama-from-container "a container cannot reach host Ollama at host.docker.internal:11434 (got '${status:-no response}')"
		return
	fi
	if grep -Fxq -- "$model" <<<"${tags//,/$'\n'}"; then
		pass ollama-from-container "reachable; OLLAMA_MODEL '$model' is present"
	else
		warn ollama-from-container "reachable, but OLLAMA_MODEL '$model' is not pulled (run: ollama pull $model)"
	fi
}

# --- run all checks (all run even after a FAIL) ------------------------------
check_docker_engine
check_docker_compose
check_docker_boot
check_disk
check_cpu
check_memory
check_time_sync
check_env_file
check_compose_config
check_ollama_from_container

exit "$had_fail"
