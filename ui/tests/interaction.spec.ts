import { test, expect } from '@playwright/test';

test.describe('AI Interaction', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the Interaction page before each test
    await page.goto('/#/interaction');
    
    // Wait for the page to load
    await expect(page.locator('h2:has-text("Ethical AI Interaction")')).toBeVisible();
  });

  test('should allow sending a message and receive a response', async ({ page }) => {
    // Enter a message
    const messageInput = page.locator('input[type="text"]');
    await messageInput.fill('Hello, how are you?');
    
    // Click the send button
    const sendButton = page.locator('button:has-text("Send")');
    await sendButton.click();
    
    // Wait for the response
    const chatMessages = page.locator('.chat-message');
    await expect(chatMessages).toHaveCount(2, { timeout: 15000 }); // User message + AI response
    
    // Verify the AI response is displayed
    const aiResponse = chatMessages.last();
    await expect(aiResponse).toContainText(/AI:/);
  });

  test('should display error for empty message', async ({ page }) => {
    // Click send without entering a message
    const sendButton = page.locator('button:has-text("Send")');
    await sendButton.click();
    
    // Verify that an error message is displayed
    const errorMessage = page.locator('.error-message');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText('Please enter a message');
  });

  test('should display conversation history', async ({ page }) => {
    // Send a message
    const messageInput = page.locator('input[type="text"]');
    await messageInput.fill('What is ethical AI?');
    
    const sendButton = page.locator('button:has-text("Send")');
    await sendButton.click();
    
    // Wait for the response
    await expect(page.locator('.chat-message')).toHaveCount(2, { timeout: 15000 });
    
    // Refresh the page
    await page.reload();
    
    // Verify the conversation history is still there
    const chatMessages = page.locator('.chat-message');
    await expect(chatMessages).toHaveCount(2);
  });
});
