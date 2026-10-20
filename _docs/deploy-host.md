# Deploy-host runbook

How to prepare the Ubuntu server that runs the ytdigest stack
(`compose.yml`, architecture §11) and the #3 transcription bake-off. Follow the
sections in order, then run `ops/host/preflight.sh` to confirm the host is
ready, then do the first deploy.

> **This host, specifically.** The target is an Ubuntu 26.10 box (Intel Core
> i7-4712MQ, 4C/8T; 16 GB RAM). The architecture names Ubuntu 26.04 LTS
> (§16.2) for Python 3.14 parity; 26.10 is fine for the Docker path because the
> services run in `python:3.14-slim` containers, so host parity holds at the
> container, not the interpreter build. This same box is also slated to run a
> k3s cluster (issue #210, "Platform rollout"). Nothing here conflicts with
> that — Docker Engine and k3s coexist — but if k3s is installed first, re-run
> `ops/host/preflight.sh` to confirm Docker's `disk`, `cpu` and `memory`
> headroom still clears the floors with k3s also resident.

Commands that change the system are marked **(sudo)**. The rest are read-only
or run as the deploy account.

---

## 1. Prerequisites and version floors

- **OS:** Ubuntu 26.10 (or 26.04 LTS), 64-bit.
- **Docker Engine ≥ 25.0** — the worker healthcheck's `start_interval` in
  `compose.yml` (#58) needs Engine 25.0+.
- **Docker Compose plugin (v2) ≥ 2.24.0** — `deploy.resources.limits` honoured
  outside Swarm (§11.5), `build.additional_contexts` with `service:`,
  `start_interval`, and `up --wait --wait-timeout` all require 2.24.0+. If a
  later Compose feature used by `compose.yml` turns out to need a higher
  version, raise this floor and cite the Compose release note.
- **Hardware floors** (checked by the preflight): ≥ 4 CPUs (so
  `TRANSCRIBER_CPUS=3.0` is accepted — Docker rejects a CPU quota above the
  host count), ≥ 12 GB RAM, ≥ 40 GB free on Docker's data root.

Legacy `docker-compose` v1 (the Python script, `docker-compose` with a hyphen)
is **not** used anywhere; install only the v2 plugin (`docker compose`, a
space).

---

## 2. Install Docker Engine and the Compose v2 plugin

From Docker's official apt repository for Ubuntu. **(sudo)** throughout.

```bash
# 2.1 base tools and keyring
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 2.2 apt source. VERSION_CODENAME comes from /etc/os-release.
. /etc/os-release
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 2.3 install engine, CLI, containerd, buildx and the compose v2 plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

> **New Ubuntu release gotcha.** If `sudo apt-get update` reports no Release
> file for `${VERSION_CODENAME}` (Docker's repo lags a brand-new Ubuntu like
> 26.10), pin the previous LTS codename Docker does publish — set
> `VERSION_CODENAME` by hand in step 2.2 to that codename — and note it here.
> Do not build Docker from source.

Enable the daemon at boot, so `restart: unless-stopped` services return after a
reboot. **(sudo)**

```bash
sudo systemctl enable --now docker.service containerd.service
systemctl is-enabled docker        # must print: enabled
docker version --format '{{.Server.Version}}'   # must be >= 25.0
docker compose version --short                   # must be >= 2.24.0
```

**Rootless Docker is not used.** `host-gateway` (how the analyzer reaches the
host's Ollama) and the fixed loopback publish for `web` behave differently under
rootless, and the stack is written for rootful Engine. Do not install rootless
Docker here.

---

## 3. Storage

Docker's images, volumes and build cache must have room. The preflight's `disk`
check wants ≥ 40 GB free on the filesystem holding Docker's data root.

```bash
docker info --format '{{.DockerRootDir}}'    # usually /var/lib/docker
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

If the root filesystem is too small, move the data root **before pulling any
image**. **(sudo)**

```bash
sudo systemctl stop docker
sudo mkdir -p /etc/docker
# edit /etc/docker/daemon.json to add: { "data-root": "/srv/docker" }
sudo rsync -aP /var/lib/docker/ /srv/docker/    # only if moving an existing root
sudo systemctl start docker
docker info --format '{{.DockerRootDir}}'       # confirm the new path
```

**Sizing** (adds up to the preflight's 40 GB default):

| Item | Approx size |
|---|---|
| Images: backend, whisper (+~2 GB over backend), web, bake-off (#76) | ~6–8 GB |
| `whisper-models` volume (~3 GB per large model; ~3 GB for the #3 bake-off set) | ~3–6 GB |
| `audio` volume (`AUDIO_MAX_GB`, default 20) | ≤ 20 GB |
| `pgdata` volume | grows with history |
| build cache | a few GB |

Reclaim space with `docker builder prune` and `docker image prune`.
**`docker compose down -v` deletes `pgdata`** — never run it on the server.

---

## 4. Access

- The **`docker` group is root-equivalent** (a member can start a container
  that mounts `/`). Add only named deploy accounts.

  ```bash
  sudo usermod -aG docker <deploy-user>     # (sudo)
  getent group docker                       # list members; audit this
  ```

- Clone the repo over a **read-only deploy key** (or HTTPS with a read-only
  token) into one fixed directory owned by the deploy account. Nobody edits the
  server checkout by hand; updates are `git pull` of `master` plus a rebuild.

---

## 5. Get the repo onto the host

```bash
sudo install -d -o <deploy-user> -g <deploy-user> /opt/ytdigest     # (sudo)
sudo -u <deploy-user> git clone git@github.com:ulyss-1/simple_video_summariser.git /opt/ytdigest
cd /opt/ytdigest
```

All later commands run from `/opt/ytdigest` as the deploy account.

---

## 6. `.env` and secrets

The server's `.env` lives in the checkout root, owned by the deploy account,
mode `600`.

```bash
cp .env.example .env
chmod 600 .env
```

Then edit `.env`:

- **`COMPOSE_FILE=compose.yml`** — add this line. It stops a bare
  `docker compose` on the host from merging the dev overlay
  `compose.override.yml` (which carries source bind mounts and
  `WHISPER_MODEL=tiny`; #59). `.env.example` must **not** set `COMPOSE_FILE`,
  or it would break the dev overlay on developer machines.
- **`POSTGRES_PASSWORD`** — a strong value using only `[A-Za-z0-9_-]` (it is
  inserted unencoded into `DATABASE_URL`; #58). Generate one with:

  ```bash
  tr -dc 'A-Za-z0-9_-' < /dev/urandom | head -c 40; echo
  ```

- **Anthropic, legacy environment mode** — only if running
  `SUMMARIZER=anthropic`: put `ANTHROPIC_API_KEY` in `.env` (mode 600), never in
  a shell profile, shell history, or on a command line. Leave
  `ANTHROPIC_API_KEY_FILE` empty and deploy with the base command below.
- **Ollama mode** — leave both Anthropic variables empty and use the base
  command. No secret file is required.
- **Anthropic, recommended file mode** — create the file outside the checkout
  as the deploy account, without putting the key in a command argument or shell
  history:

  ```bash
  install -d -m 700 "$HOME/.config/ytdigest"
  umask 077
  read -rsp 'Anthropic API key: ' ANTHROPIC_KEY; printf '\n'
  printf '%s' "$ANTHROPIC_KEY" > "$HOME/.config/ytdigest/anthropic_api_key"
  unset ANTHROPIC_KEY
  chmod 600 "$HOME/.config/ytdigest/anthropic_api_key"
  ```

  The deploy account must own the file. Remove the legacy
  `ANTHROPIC_API_KEY` line from `.env`, then set only the non-secret path:

  ```dotenv
  ANTHROPIC_API_KEY_FILE=/home/<deploy-user>/.config/ytdigest/anthropic_api_key
  ```

  A missing variable makes Compose configuration fail; a missing file makes
  deployment fail. Do not create a dummy or empty fallback.

`.env` is gitignored; never commit it.

---

## 7. Networking and egress

Only the reverse proxy's ports are open inbound (the proxy itself is #141);
`web` stays on `127.0.0.1:8080` (§11.4, D13). Outbound HTTPS the host and
containers need:

| Destination | For |
|---|---|
| `www.youtube.com`, `*.googlevideo.com` | video metadata and audio (yt-dlp) |
| `huggingface.co` + its CDN | Whisper models |
| `pypi.org`, `files.pythonhosted.org` | image builds |
| `deb.debian.org` | image builds (Debian base) |
| `registry.npmjs.org` | web image build |
| `registry-1.docker.io` | base images |
| `api.anthropic.com` | only when `SUMMARIZER=anthropic` |

---

## 8. Ollama

The analyzer reaches the host's Ollama at `http://host.docker.internal:11434`
(`extra_hosts: host.docker.internal:host-gateway`). On Linux, Ollama defaults
to `127.0.0.1:11434`, which containers **cannot** reach through host-gateway.
Bind it to an address containers can reach.

**Default choice (used here): bind to the Docker bridge gateway**
(`docker0`, `172.17.0.1`). It is not exposed on any external interface, and
needs no firewall rule. Install the committed drop-in. **(sudo)**

```bash
sudo install -D -m 644 ops/host/ollama.override.conf \
  /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Trade-off: Ollama is no longer on `127.0.0.1`, so host-local clients must point
at the bridge address, and Ollama must start after `docker.service` (the
drop-in sets `After=docker.service`). Pull the model:

```bash
OLLAMA_HOST=172.17.0.1:11434 ollama pull "${OLLAMA_MODEL:-qwen3.5:4b}"
```

**Alternative: bind to `0.0.0.0` with a firewall rule** that allows 11434 only
from Docker's bridge address pools and denies it everywhere else. This keeps
Ollama on `127.0.0.1` for host clients but needs the firewall rule to be
correct and maintained. If you use it, set `OLLAMA_HOST=0.0.0.0:11434` and add
the rule; note that an active `ufw` default-deny also blocks
container-to-host traffic unless the Docker bridge ranges are allowed. The
bridge-gateway default above avoids both.

---

## 9. UID/GID for host-mounted output

All project images run as `app`, UID/GID `10001` (`ops/Dockerfile.backend`,
#55). Named volumes need no host action — Docker sets their ownership. Only a
**host directory bind-mounted for output** (e.g. #76's bake-off report dir)
needs creating with that owner. **(sudo)**

```bash
sudo install -d -o 10001 -g 10001 /srv/ytdigest-bakeoff
```

To read that output as the deploy account without running containers as root,
add the deploy account to group `10001`, or `chmod g+r` the files.

---

## 10. Run the preflight

From the checkout root, as the deploy account:

```bash
ops/host/preflight.sh
```

It prints one `PASS | WARN | FAIL <check>: <detail>` line per requirement and
exits 0 when nothing FAILs (WARNs allowed), 1 when any check FAILs, 2 on a
usage error. It is read-only: it installs nothing, edits no config, never runs
`sudo`, and never prints a secret. Fix every FAIL before deploying; review each
WARN. `ops/host/preflight.sh --help` lists all checks and their threshold
variables.

---

## 11. First deploy

Migrations run as a standalone one-shot, never automatically on startup
(AGENTS.md; concurrent replicas racing a migration can corrupt the schema).

Ollama or legacy Anthropic environment mode:

```bash
docker compose -f compose.yml run --rm migrate
# The production path never merges the dev overlay.
docker compose -f compose.yml up -d --build
```

Anthropic file mode instead uses both files for migration and startup:

```bash
docker compose -f compose.yml -f compose.anthropic.yml run --rm migrate
docker compose -f compose.yml -f compose.anthropic.yml up -d --build
```

Before and after an Anthropic file-mode deployment, verify that neither the
resolved model nor the analyzer container environment carries the key. These
commands print names and paths only; inspect the output and do not paste the
key into either command:

```bash
docker compose -f compose.yml -f compose.anthropic.yml config --format json \
  | grep -E 'ANTHROPIC_API_KEY|anthropic_api_key'
docker inspect ytdigest-analyzer-1 --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep ANTHROPIC
```

The resolved config must contain only `ANTHROPIC_API_KEY_FILE` and the secret
path/source, never `ANTHROPIC_API_KEY`. The inspect output must contain only
`ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key`. Container names can
differ; obtain the analyzer name with `docker compose ps analyzer`.

Sanity checks:

```bash
# no dev overlay leaked in: no bind mounts, no tiny model
docker compose config | grep -E 'type: bind|WHISPER_MODEL:\s*"?tiny"?' || echo "clean"

docker compose ps          # services healthy
```

`web` is published on `127.0.0.1:8080` only; put it behind the host reverse
proxy (#141) for TLS. Updates are `git pull` of `master`, then
`docker compose run --rm migrate` (if there is a new revision) and
`docker compose -f compose.yml up -d --build`.
