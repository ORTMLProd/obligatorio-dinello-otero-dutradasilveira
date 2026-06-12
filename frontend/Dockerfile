# ── Build stage: compile the React app to static assets ──────────────────────
# node:22-alpine (LTS) for reproducibility, independent of the host Node version.
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# ── Serve stage: nginx serves the static build and reverse-proxies /api ───────
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
