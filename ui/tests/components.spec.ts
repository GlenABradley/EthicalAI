import { test, expect } from '@playwright/test';

test.describe('Component Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000');
  });

  test('renders all navigation tabs', async ({ page }) => {
    const tabs = ['Status', 'Axes', 'Analyze', 'Interaction'];
    
    for (const tab of tabs) {
      const tabElement = page.locator(`button:has-text("${tab}")`);
      await expect(tabElement).toBeVisible();
    }
  });

  test('navigates between tabs', async ({ page }) => {
    const tabTests = [
      { name: 'Status', heading: 'Status' },
      { name: 'Axes', heading: 'Axes' },
      { name: 'Analyze', heading: 'Analyze' },
      { name: 'Interaction', heading: 'Analyze' },
    ];

    for (const { name, heading } of tabTests) {
      // Click the tab
      const tab = page.locator(`button:has-text("${name}")`);
      await tab.click();
      
      // Wait for navigation and content to load
      await page.waitForLoadState('networkidle');
      
      // Verify the heading is visible
      const headingElement = page.locator(`h2:has-text("${heading}")`).first();
      await expect(headingElement).toBeVisible();
      
      // Verify the tab is active
      await expect(tab).toHaveCSS('background-color', 'rgb(238, 238, 255)');
    }
  });

  test('analyzes text in the Analyze tab', async ({ page }) => {
    // Navigate to Analyze tab
    await page.click('button:has-text("Analyze")');
    
    // Enter test text
    const testText = 'This is a test sentence about ethics.';
    await page.fill('textarea', testText);
    
    // Click analyze button
    const analyzeButton = page.locator('button:has-text("Analyze")').nth(1);
    await analyzeButton.click();
    
    // Wait for results
    await page.waitForSelector('.analysis-result', { state: 'visible' });
    
    // Verify results are displayed
    const results = await page.locator('.analysis-result').all();
    expect(results.length).toBeGreaterThan(0);
  });

  test('interacts with the chat interface', async ({ page }) => {
    // Navigate to Interaction tab
    await page.click('button:has-text("Interaction")');
    
    // Enter a message
    const testMessage = 'Hello, how are you?';
    await page.fill('input[type="text"]', testMessage);
    
    // Send the message
    await page.click('button:has-text("Send")');
    
    // Wait for response
    await page.waitForSelector('.message-bot', { state: 'visible' });
    
    // Verify the message was sent and received
    const messages = await page.locator('.message').all();
    expect(messages.length).toBeGreaterThanOrEqual(2); // At least user message and bot response
  });
});
