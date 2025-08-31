describe('API Integration', () => {
  const API_BASE = Cypress.env('API_BASE') || 'http://localhost:8000';

  it('health check endpoint is accessible', () => {
    cy.request(`${API_BASE}/health/ready`).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property('status');
    });
  });

  it('can fetch axes', () => {
    cy.request(`${API_BASE}/v1/axes`).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.be.an('array');
    });
  });

  // Add more API tests as needed
});
