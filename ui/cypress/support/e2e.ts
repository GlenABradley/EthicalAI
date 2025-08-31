// ***********************************************************
// This example support/e2e.ts is processed and
// loaded automatically before your test files.
// ***********************************************************

// Import commands.js using ES2015 syntax:
import '@testing-library/cypress/add-commands';

// Alternatively you can use CommonJS syntax:
// require('./commands')

// Add global error handling
Cypress.on('uncaught:exception', (err) => {
  console.error('Uncaught exception:', err);
  // returning false here prevents Cypress from failing the test
  return false;
});
