import { test, expect } from '@playwright/test';

test.describe('Axes Management', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the Axes page before each test
    await page.goto('/#/axes');
    
    // Wait for the page to load
    await expect(page.locator('h2:has-text("Axes Management")')).toBeVisible();
  });

  test('should display available axes', async ({ page }) => {
    // Wait for axes to load
    const axesList = page.locator('.axes-list');
    await expect(axesList).toBeVisible({ timeout: 10000 });
    
    // Verify that some axes are displayed
    const axesItems = page.locator('.axis-item');
    const count = await axesItems.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should allow activating an axis', async ({ page }) => {
    // Wait for axes to load
    const axesItems = page.locator('.axis-item');
    await expect(axesItems.first()).toBeVisible({ timeout: 10000 });
    
    // Click the activate button on the first axis
    const activateButton = axesItems.first().locator('button:has-text("Activate")');
    await activateButton.click();
    
    // Verify that the axis is now active
    const activeBadge = axesItems.first().locator('.active-badge');
    await expect(activeBadge).toBeVisible();
    await expect(activeBadge).toContainText('Active');
  });

  test('should allow deactivating an axis', async ({ page }) => {
    // Wait for axes to load
    const axesItems = page.locator('.axis-item');
    await expect(axesItems.first()).toBeVisible({ timeout: 10000 });
    
    // First activate the axis if it's not already active
    const activateButton = axesItems.first().locator('button:has-text("Activate")');
    if (await activateButton.isVisible()) {
      await activateButton.click();
    }
    
    // Now deactivate it
    const deactivateButton = axesItems.first().locator('button:has-text("Deactivate")');
    await expect(deactivateButton).toBeVisible();
    await deactivateButton.click();
    
    // Verify that the axis is now inactive
    const activeBadge = axesItems.first().locator('.active-badge');
    await expect(activeBadge).not.toBeVisible();
  });
});
