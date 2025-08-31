describe('EthicalAI Application', () => {
  beforeEach(() => {
    // Visit the app before each test
    cy.visit('/');
  });

  it('successfully loads', () => {
    // Check if the app title is visible
    cy.contains('h1', 'EthicalAI').should('be.visible');
    
    // Check if the main navigation is visible
    cy.get('header').should('be.visible');
    cy.get('main').should('be.visible');
  });

  it('can navigate between tabs', () => {
    // Test navigation to each tab
    const tabs = ['status', 'axes', 'analyze', 'interaction'];
    
    tabs.forEach(tab => {
      cy.contains('button', tab).click();
      // Check if the URL contains the tab name
      cy.url().should('include', `#/${tab}`);
    });
  });
});
