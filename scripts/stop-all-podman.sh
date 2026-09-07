#!/bin/bash
# Stops every model-serving profile in compose/docker-compose.yml. All profiles
# bind port 8098, so a launcher script sources this and calls stop_all before
# it starts its own profile.

STOP_ALL_PROFILES=(dsv4 dsv4backport qwen qwen8gpu qwenawq glm53flash glm53int4)

stop_all() {
  local compose_dir
  compose_dir="$(dirname "${BASH_SOURCE[0]}")/../compose"
  ( cd "$compose_dir" || exit 1
    for profile in "${STOP_ALL_PROFILES[@]}"; do
      podman compose --profile "$profile" down >/dev/null 2>&1
    done
  )
}
