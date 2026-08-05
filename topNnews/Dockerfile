FROM node:22-alpine

ENV NODE_ENV=production
WORKDIR /app

# No dependencies to install — the app is stdlib-only, so there is no npm install step.
COPY package.json server.js ./
COPY src ./src
COPY public ./public

EXPOSE 3000
CMD ["node", "server.js"]
