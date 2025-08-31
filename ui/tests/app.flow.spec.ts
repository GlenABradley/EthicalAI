import { test, expect } from '@playwright/test';

test.describe('EthicalAI Application Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/EthicalAI/);
  });

  test('should navigate through all tabs', async ({ page }) => {
    // Helper function to test tab navigation
    const testTab = async (tabName: string, expectedUrl: string) => {
      await page.click(`button:has-text("${tabName}")`);
      await expect(page).toHaveURL(new RegExp(expectedUrl));
      await expect(page.locator('main')).toBeVisible();
    };

    // Test each tab
    await testTab('status', '.*');
    await testTab('axes', '.*#/axes');
    await testTab('analyze', '.*#/analyze');
    await testTab('interaction', '.*#/interaction');
  });

  test('should display API status', async ({ page }) => {
    await page.click('button:has-text("status")');
    await expect(page.locator('text=API Status')).toBeVisible();
  });

  test('should display axes management', async ({ page }) => {
    await page.click('button:has-text("axes")');
    await expect(page.locator('h2:has-text("Axes Management")')).toBeVisible();
  });

  test('should display analyze interface', async ({ page }) => {
    await page.click('button:has-text("analyze")');
    await expect(page.locator('h2:has-text("Analyze Text")')).toBeVisible();
  });

  test('should display interaction interface', async ({ page }) => {
    await page.click('button:has-text("interaction")');
    await expect(page.locator('h2:has-text("Ethical AI Interaction")')).toBeVisible();
  });
});
