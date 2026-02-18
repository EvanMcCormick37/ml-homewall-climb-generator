# Builder Stage
FROM node:20-slim AS builder

WORKDIR /app

COPY package.json package-lock.json ./
run npm ci

COPY . .

ARG VITE_API_URL=/api/v1
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Caddy Serving Stage
FROM caddy:2-alpine

COPY --from=builder /app/dist /srv

COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 80

