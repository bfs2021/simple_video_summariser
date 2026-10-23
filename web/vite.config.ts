import babel from '@rolldown/plugin-babel'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { API_PREFIX, rewriteApiPath } from './src/devProxy.ts'
import { TEST_TIME_ZONE } from './src/test/timeZone.ts'

// Non-VITE_ on purpose: it must never reach the client bundle.
const apiProxyTarget = process.env.API_PROXY_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  base: '/',
  plugins: [react(), babel({ presets: [reactCompilerPreset()] })],
  build: { sourcemap: false },
  server: {
    proxy: {
      [API_PREFIX]: { target: apiProxyTarget, rewrite: rewriteApiPath },
    },
  },
  test: {
    // Pinned here, not in a developer's shell, so UTC-only date handling fails.
    env: { TZ: TEST_TIME_ZONE },
    // Two mutually exclusive projects. `extends: true` reuses the plugin list
    // above, so the React Compiler is configured in exactly one place.
    projects: [
      {
        extends: true,
        test: {
          name: 'node',
          environment: 'node',
          include: ['src/**/*.test.{ts,tsx}'],
          exclude: ['**/node_modules/**', 'src/**/*.dom.test.{ts,tsx}'],
        },
      },
      {
        extends: true,
        test: {
          name: 'jsdom',
          environment: 'jsdom',
          include: ['src/**/*.dom.test.{ts,tsx}'],
          setupFiles: ['./src/test/setup.ts'],
        },
      },
    ],
  },
})
