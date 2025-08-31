import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import type { RequestHandler } from 'msw';

// Define the base URL for the API
const API_BASE = 'http://localhost:8000';

// Mock data
const mockHealth = { status: 'ok' };

const mockActiveAxes = {
  pack_id: 'test-pack',
  axes: ['test-axis-1', 'test-axis-2']
};

const mockEvalResult = {
  proof: { decision: 'allow', reasoning: 'Test reasoning' },
  spans: [
    { text: 'test', start: 0, end: 4, scores: [0.5, 0.5] }
  ]
};

const mockResponse = {
  final: 'Test response',
  proof: { decision: 'allow', reasoning: 'Test reasoning' },
  alternatives: [
    { text: 'Alternative response 1' },
    { text: 'Alternative response 2' }
  ]
};

// Define request handlers
const handlers: RequestHandler[] = [
  // Health check
  http.get(`${API_BASE}/health/ready`, () => {
    return HttpResponse.json(mockHealth);
  }),
  
  // Build axes
  http.post(`${API_BASE}/v1/axes/build`, async ({ request }) => {
    const { names } = await request.json() as { names: string[] };
    return HttpResponse.json({
      pack_id: 'test-pack',
      axes: names
    });
  }),
  
  // Activate axes
  http.post(`${API_BASE}/v1/axes/activate`, () => {
    return HttpResponse.json({ ok: true });
  }),
  
  // Get active axes
  http.get(`${API_BASE}/v1/axes/active`, () => {
    return HttpResponse.json(mockActiveAxes);
  }),
  
  // Evaluate text
  http.post(`${API_BASE}/v1/eval/text`, async () => {
    return HttpResponse.json(mockEvalResult);
  }),
  
  // Get response
  http.post(`${API_BASE}/v1/interaction/respond`, async ({ request }) => {
    const { prompt } = await request.json() as { prompt: string };
    if (!prompt) return new HttpResponse(null, { status: 400 });
    return HttpResponse.json(mockResponse);
  }),
];

export const server = setupServer(...handlers);

// Test lifecycle methods are provided by the test runner
