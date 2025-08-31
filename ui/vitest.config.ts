/// <reference types="vitest" />
/// <reference types="vite/client" />

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'url';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    testTimeout: 10000,
    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],
    exclude: ['node_modules', 'dist', '.idea', '.git', '.cache'],
    coverage: {
      enabled: true,
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/setupTests.ts',
        '**/*.d.ts',
        '**/*.test.{js,ts,jsx,tsx}',
        '**/*.spec.{js,ts,jsx,tsx}',
      ],
    },
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
    server: {
      deps: {
        inline: ['@testing-library/user-event'],
      },
    },
  },
});
