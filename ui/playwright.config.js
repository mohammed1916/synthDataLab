import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests',
  timeout: 30 * 1000,
  expect: {
    timeout: 5000,
  },
  use: {
    baseURL: 'http://127.0.0.1:5175',
    headless: true,
    viewport: { width: 1280, height: 720 },
  },
  webServer: {
    command: 'npm run dev -- --port 5175 --host 127.0.0.1',
    port: 5175,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
