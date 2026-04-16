FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production \
	PORT=80 \
	API_TARGET_HOST=backend \
	API_TARGET_PORT=8001

COPY --from=build /app/dist ./dist
COPY server.js ./server.js

EXPOSE 80

CMD ["node", "server.js"]
