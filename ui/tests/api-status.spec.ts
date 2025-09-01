import { test, expect } from '@playwright/test';

test.describe('API Status', () => {
  test('should display API status correctly', async ({ page }) => {
    // Navigate to the Status page
    await page.goto('/#/status');
    
    // Wait for the API status to be loaded
    const statusElement = page.locator('[data-testid="api-status"]');
    await expect(statusElement).toBeVisible({ timeout: 10000 });
    
    // Check if the status is displayed (either "Online" or "Offline")
    const statusText = await statusElement.textContent();
    expect(statusText).toMatch(/(Online|Offline)/);
    
    // If there's a timestamp, verify it's displayed
    const timestampElement = page.locator('[data-testid="api-timestamp"]');
    if (await timestampElement.isVisible()) {
      const timestamp = await timestampElement.textContent();
      expect(timestamp).toBeTruthy();
    }
  });
});
