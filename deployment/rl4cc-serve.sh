#!/bin/bash

# ------------------------------------------------------------------
# Run optional bootstrap modules
# ------------------------------------------------------------------
if [ -n "$APP_BOOTSTRAP_MODULES" ]; then
  echo "Running bootstrap modules: $APP_BOOTSTRAP_MODULES"

  # Split comma-separated list
  IFS=',' read -ra MODULES << "$APP_BOOTSTRAP_MODULES"

  for module in "${MODULES[@]}"; do
    module_trimmed=$(echo "$module" | xargs)  # trim spaces

    if [ -n "$module_trimmed" ]; then
      echo "Importing module: $module_trimmed"
      python -c "import ${module_trimmed}"
    fi
  done
fi

# ------------------------------------------------------------------
# Start FastAPI app
# ------------------------------------------------------------------
exec uvicorn RL4CC.serve.rl_agent_server:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
