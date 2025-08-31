import { test, expect } from '@playwright/test';

test.describe('Analyze Page', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the Analyze page before each test
    await page.goto('/#/analyze');
    
    // Wait for the page to load
    await expect(page.locator('h2:has-text("Analyze Text")')).toBeVisible();
  });

  test('should analyze text and display results', async ({ page }) => {
    // Enter text to analyze
    const textInput = page.locator('textarea');
    await textInput.fill('This is a test sentence to analyze for ethical considerations.');
    
    // Click the analyze button
    const analyzeButton = page.locator('button:has-text("Analyze")');
    await analyzeButton.click();
    
    // Wait for analysis results
    const results = page.locator('.analysis-results');
    await expect(results).toBeVisible({ timeout: 10000 });
    
    // Verify that some analysis output is displayed
    const resultText = await results.textContent();
    expect(resultText).toBeTruthy();
  });

  test('should display error for empty input', async ({ page }) => {
    // Click the analyze button without entering any text
    const analyzeButton = page.locator('button:has-text("Analyze")');
    await analyzeButton.click();
    
    // Verify that an error message is displayed
    const errorMessage = page.locator('.error-message');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText('Please enter some text to analyze');
  });
});
