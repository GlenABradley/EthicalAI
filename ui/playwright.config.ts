import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: false, // Run tests in sequence to avoid port conflicts
  forbidOnly: false, // Disable CI check to avoid TypeScript errors
  retries: 1, // Retry failed tests once
  workers: 1, // Use a single worker to avoid port conflicts
  reporter: [
    ['html', { open: 'never' }],
    ['list'],
    ['line']
  ],
  timeout: 30000, // Global timeout for all tests
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'on-first-retry',
    testIdAttribute: 'data-testid',
    // Enable debug logging
    launchOptions: {
      slowMo: 100,
      headless: false,
      devtools: true
    },
    // Configure context options
    contextOptions: {
      ignoreHTTPSErrors: true,
      viewport: { width: 1280, height: 800 },
    },
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
  // Use the existing development server
  webServer: {
    command: 'echo Using existing server on port 3000',
    port: 3000,
    reuseExistingServer: true, // Reuse the existing server
    timeout: 5000,
    stderr: 'pipe',
    stdout: 'pipe'
  },
  expect: {
    timeout: 10000, // Increase timeout for assertions
  },
  // Configure global setup/teardown if needed
  // globalSetup: require.resolve('./tests/global-setup'),
  // globalTeardown: require.resolve('./tests/global-teardown'),
});
