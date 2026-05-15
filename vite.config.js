import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const webPort = Number(env.VITE_PORT || 5178);
  const apiPort = Number(env.VITE_API_PORT || 5050);

  return {
    plugins: [react()],
    server: {
      port: webPort,
      strictPort: true,
      host: true,
      open: true,
      proxy: {
        '/api': {
          target: `http://127.0.0.1:${apiPort}`,
          changeOrigin: true,
          timeout: 600000,
          proxyTimeout: 600000,
        },
        '/outputs': {
          target: `http://127.0.0.1:${apiPort}`,
          changeOrigin: true,
        },
      },
    },
  };
});
