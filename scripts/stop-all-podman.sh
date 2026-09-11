#!/bin/bash
# Stops every model-serving profile in compose/docker-compose.yml. All profiles
# bind port 8098, so a launcher script sources this and calls stop_all before
# it starts its own profile.

STOP_ALL_PROFILES=(dsv4 dsv4backport dsv41 dsv416 dsv416pp dsv418 qwen qwen8gpu qwenawq glm53flash glm53flash6gpu glm53int4 glm53int48gpu glm53mix8gpu)

stop_all() {
  local compose_dir
  compose_dir="$(dirname "${BASH_SOURCE[0]}")/../compose"
  ( cd "$compose_dir" || exit 1
    for profile in "${STOP_ALL_PROFILES[@]}"; do
      podman compose --profile "$profile" down >/dev/null 2>&1
    done
  )
}
