#!/bin/sh

# Check if the mounted volume directory is empty
if [ -z "$(ls -A /app/data)" ]; then
    echo "Staging data into empty persistent volume..."
    # Use cp -a to copy all contents, including hidden files, while preserving permissions
    cp -a /app/data-staging/. /app/data/
    rm -rf /app/data-staging
    echo "Data initialization complete."
else
    echo "Persistent volume already contains data. Skipping initialization."
fi

# Hand over control to the main CMD process (CRITICAL for graceful shutdowns)
exec "$@"