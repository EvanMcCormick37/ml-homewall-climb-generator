#!/bin/bash

# 1. Start Docker Desktop
echo "🚀 Starting Docker Engine..."
/c/Program\ Files/Docker/Docker/Docker\ Desktop.exe &

# Wait for Docker to be responsive
echo "Waiting for Docker to initialize..."
until docker info > /dev/null 2>&1; do
  echo -n "."
  sleep 2
done
echo -e "\n✅ Docker is ready!"

# 2. Build and Push Backend
echo "📦 Building Backend..."
cd ./climb-backend
docker build -t evanmccormick/climb-backend:latest .
docker push evanmccormick/climb-backend:latest
cd ..

# 3. Build and Push Frontend
echo "📦 Building Frontend..."
cd ./climb-frontend
docker build -t evanmccormick/climb-frontend:latest .
docker push evanmccormick/climb-frontend:latest
cd ..

# 4. Shut down Docker Desktop
echo "🛑 Shutting down Docker..."
/c/Program\ Files/Docker/Docker/resources/bin/docker-compose.exe version > /dev/null # Ensuring path access
taskkill //IM "Docker Desktop.exe" //F

echo "🎉 All done!"