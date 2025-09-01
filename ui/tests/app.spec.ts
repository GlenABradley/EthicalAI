import { test, expect } from '@playwright/test';

test.describe('EthicalAI Application', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app before each test
    await page.goto('http://localhost:5173');
  });

  test('has title', async ({ page }) => {
    // Expect the title to contain "EthicalAI"
    await expect(page).toHaveTitle(/EthicalAI/);
  });

  test('navigation works', async ({ page }) => {
    // Test navigation to each tab
    const tabs = ['status', 'axes', 'analyze', 'interaction'];
    
    for (const tab of tabs) {
      // Click the tab button
      await page.click(`button:has-text("${tab}")`);
      
      // Verify the URL contains the tab name
      await expect(page).toHaveURL(new RegExp(`#/${tab}`));
      
      // Verify the main content is visible
      await expect(page.locator('main')).toBeVisible();
    }
  });
});
