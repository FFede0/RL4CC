#!/bin/sh

exec uvicorn RL4CC.serve.rl_agent_server:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
