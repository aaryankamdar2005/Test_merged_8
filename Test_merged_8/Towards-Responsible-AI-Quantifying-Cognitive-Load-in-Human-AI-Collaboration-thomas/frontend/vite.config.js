import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const backendUrl = env.BACKEND_URL || 'http://127.0.0.1:8002';
  
  return {
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
          secure: false,
          ws: true
        }
      }
    },
    define: {
      'import.meta.env.BACKEND_URL': JSON.stringify(backendUrl)
    },
    build: {
      outDir: '../dist',
      emptyOutDir: true
    }
  };
});
