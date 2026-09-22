import { defineConfig } from 'vite';
export default defineConfig({
  server: { port: 5173, strictPort: true, proxy: {
    '/api/events': { target: 'ws://127.0.0.1:8787', ws: true },
    '/api': { target: 'http://127.0.0.1:8787', changeOrigin: false }
  } },
  build: { target: 'es2022' }
});
