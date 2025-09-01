import { server } from '../__mocks__/server';
import { api } from '../lib/api';
import { http, HttpResponse } from 'msw';
import { setupWorker } from 'msw/browser';

// Enable API mocking before tests.
beforeAll(() => {
  // Add request logging
  server.events.on('request:start', ({ request }) => {
    console.log('MSW intercepted:', request.method, request.url);
  });
  
  server.events.on('request:match', ({ request }) => {
    console.log('MSW handled:', request.method, request.url);
  });
  
  server.events.on('request:unhandled', ({ request }) => {
    console.log('MSW unhandled:', request.method, request.url);
  });
  
  return server.listen({ onUnhandledRequest: 'error' });
});

// Reset any runtime request handlers we may add during the tests.
afterEach(() => server.resetHandlers());

// Disable API mocking after the tests are done.
afterAll(() => server.close());

describe('API Client', () => {

  describe('health', () => {
    it('should check API health', async () => {
      const health = await api.health();
      expect(health).toHaveProperty('status');
      expect(health.status).toBe('ok');
    });
  });

  describe('axes', () => {
    beforeEach(() => server.resetHandlers());

    afterEach(() => server.resetHandlers());

    it('should build axes', async () => {
      const result = await api.axes.build(
        ['test-axis-1', 'test-axis-2'],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        { description: 'test' }
      );
      expect(result).toHaveProperty('pack_id', 'test-pack');
      expect(result).toHaveProperty('axes');
      expect(result.axes).toContain('test-axis-1');
      expect(result.axes).toContain('test-axis-2');
    });

    it('should activate axes', async () => {
      const result = await api.axes.activate('test-pack');
      expect(result).toHaveProperty('ok', true);
    });

    it('should get active axes', async () => {
      const result = await api.axes.active();
      expect(result).toHaveProperty('pack_id');
      expect(result).toHaveProperty('axes');
    });
  });

  describe('evalText', () => {
    it('should evaluate text', async () => {
      const result = await api.evalText('test text');
      expect(result).toHaveProperty('proof');
      expect(result).toHaveProperty('spans');
    });
  });

  describe('respond', () => {
    it('should get a response', async () => {
      const result = await api.respond('test prompt');
      expect(result).toHaveProperty('final');
      expect(result).toHaveProperty('proof');
      expect(result).toHaveProperty('alternatives');
    });
  });
});
