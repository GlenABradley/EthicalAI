import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: false, // Run tests in sequence to avoid port conflicts
  forbidOnly: !!process.env.CI,
  retries: 1, // Retry failed tests once
  workers: 1, // Use a single worker to avoid port conflicts
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
      },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000, // Increase timeout to 2 minutes
    stderr: 'pipe',
    stdout: 'pipe',
  },
  expect: {
    timeout: 10000, // Increase timeout for assertions
  },
});
