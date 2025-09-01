import { test, expect, Page } from '@playwright/test';

const BASE_URL = 'http://localhost:3000';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the homepage before each test
    await page.goto(BASE_URL);
    // Wait for the main content to be visible
    await page.waitForSelector('main, #root', { state: 'visible', timeout: 10000 });
  });

  const testNavigation = async (page: Page, tabName: string, expectedHeading: string) => {
    console.log(`Testing navigation to ${tabName} page...`);
    
    try {
      // Find and click the tab button
      const tabButton = page.locator(`button:has-text("${tabName}")`).first();
      await expect(tabButton, `${tabName} tab button should be visible`).toBeVisible({ timeout: 10000 });
      
      console.log(`Clicking ${tabName} tab...`);
      await tabButton.click();
      
      // Wait for the page content to be visible
      console.log(`Looking for heading: ${expectedHeading}`);
      const content = page.locator(`h2:has-text("${expectedHeading}")`).first();
      await expect(content, `Heading "${expectedHeading}" should be visible`).toBeVisible({ timeout: 15000 });
      
      // Verify the tab is active
      await expect(tabButton).toHaveCSS('background-color', 'rgb(238, 238, 255)');
      
      console.log(`Successfully navigated to ${tabName} page`);
    } catch (error) {
      console.error(`Error during navigation to ${tabName}:`, error);
      console.log('Current page URL:', page.url());
      console.log('Page content:', await page.content());
      await page.screenshot({ path: `test-results/navigation-failure-${tabName.toLowerCase()}.png` });
      throw error;
    }
  };

  test('should navigate to Status page', async ({ page }) => {
    await testNavigation(page, 'status', 'Status');
  });

  test('should navigate to Axes page', async ({ page }) => {
    await testNavigation(page, 'axes', 'Axes');
  });

  test('should navigate to Analyze page', async ({ page }) => {
    await testNavigation(page, 'analyze', 'Analyze');
  });

  test('should navigate to Interaction page', async ({ page }) => {
    await testNavigation(page, 'interaction', 'Analyze');
  });
});
