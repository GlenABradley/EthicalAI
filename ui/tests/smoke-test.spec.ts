import { test, expect } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

test('basic test', async ({ page, browser }) => {
  // Set a longer timeout for this test
  test.setTimeout(120000);
  
  console.log('Starting test...');
  
  try {
    // Enable request/response logging
    page.on('request', request => console.log('>>', request.method(), request.url()));
    page.on('response', response => console.log('<<', response.status(), response.url()));
    
    // Navigate to the application with a longer timeout
    const url = 'http://localhost:3000';
    console.log(`Navigating to ${url}...`);
    
    try {
      console.log('Before page.goto');
      const response = await page.goto(url, { 
        waitUntil: 'domcontentloaded',
        timeout: 60000
      });
      
      if (!response) {
        throw new Error('No response from page.goto');
      }
      
      console.log('Navigation completed, status:', response.status());
      console.log('Response URL:', response.url());
      
      if (!response.ok()) {
        console.error('Navigation failed with status:', response.status());
        console.error('Status text:', response.statusText());
        const body = await response.text().catch(() => 'Could not get response body');
        console.error('Response body (first 500 chars):', body.slice(0, 500));
        throw new Error(`Navigation failed with status ${response.status()}: ${response.statusText()}`);
      }
    } catch (error) {
      console.error('Error during navigation:', error);
      const pageContent = await page.content().catch(() => 'Could not get page content');
      console.log('Page content (first 500 chars):', pageContent.slice(0, 500));
      throw error;
    }
    
    // Wait for a specific element to be visible instead of networkidle
    console.log('Waiting for main content...');
    await page.waitForSelector('main, #root, [data-testid="app-container"]', { 
      state: 'attached',
      timeout: 30000 
    });
    
    // Get the page content for debugging
    const content = await page.content();
    console.log('Page content length:', content.length);
    
    // Take a screenshot for debugging
    console.log('Taking screenshot...');
    await page.screenshot({ path: 'smoke-test-screenshot.png' });
    
    // Check if the page has loaded by looking for a known element
    console.log('Getting page title...');
    const title = await page.title();
    console.log('Page title:', title);
    
    // Basic assertion to verify the page is responsive
    console.log('Verifying page title...');
    await expect(page).toHaveTitle(/EthicalAI|React App/);
    
    console.log('Test completed successfully!');
  } catch (error) {
    console.error('Test failed:', error);
    // Take a screenshot on failure
    await page.screenshot({ path: 'smoke-test-failure.png' });
    
    // Get browser logs
    console.log('\nBrowser console logs:');
    const logs = await page.evaluate(() => {
      return Array.from(document.getElementsByTagName('script')).map(script => script.src);
    });
    console.log('Scripts loaded:', logs);
    
    throw error;
  }
});
