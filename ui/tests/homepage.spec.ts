import { test, expect } from '@playwright/test';

test.describe('Homepage', () => {
  test('should load the homepage', async ({ page }) => {
    // Navigate to the homepage
    await page.goto('/');
    
    // Verify the page title
    await expect(page).toHaveTitle(/EthicalAI/);
    
    // Verify the main content is visible
    const mainContent = page.locator('main');
    await expect(mainContent).toBeVisible();
  });
});
