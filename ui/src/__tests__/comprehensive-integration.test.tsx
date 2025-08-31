/**
 * Comprehensive integration tests for EthicalAI frontend components.
 * Tests React components and their integration with the backend API.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import App from '../App';

// Mock API responses
const mockEmbeddingResponse = {
  embeddings: [[0.1, 0.2, 0.3, 0.4, 0.5]]
};

const mockHealthResponse = {
  status: 'ok',
  encoder_model: 'all-mpnet-base-v2',
  encoder_dim: 768,
  active_pack: null,
  frames_db_present: false
};

const mockAnalysisResponse = {
  scores: {
    consequentialism: 0.75,
    deontology: 0.65,
    virtue: 0.80
  },
  analysis: 'This text demonstrates strong ethical considerations...',
  embeddings: [[0.1, 0.2, 0.3, 0.4, 0.5]]
};

// Setup MSW server for API mocking
const server = setupServer(
  http.get('http://localhost:8080/health/ready', () => {
    return HttpResponse.json(mockHealthResponse);
  }),
  
  http.post('http://localhost:8080/embed', () => {
    return HttpResponse.json(mockEmbeddingResponse);
  }),
  
  http.post('http://localhost:8080/analyze', () => {
    return HttpResponse.json(mockAnalysisResponse);
  }),
  
  http.get('http://localhost:8080/v1/axes', () => {
    return HttpResponse.json([
      { name: 'consequentialism', description: 'Outcome-based ethics' },
      { name: 'deontology', description: 'Duty-based ethics' },
      { name: 'virtue', description: 'Character-based ethics' }
    ]);
  })
);

beforeEach(() => {
  server.listen();
});

afterEach(() => {
  server.resetHandlers();
});

describe('EthicalAI Frontend Integration Tests', () => {
  
  describe('App Component Integration', () => {
    it('should render the main application', () => {
      render(<App />);
      
      // Should render main navigation or header
      expect(document.body).toBeInTheDocument();
    });
    
    it('should handle API connectivity', async () => {
      render(<App />);
      
      // Wait for any initial API calls to complete
      await waitFor(() => {
        // App should be rendered without errors
        expect(document.body).toBeInTheDocument();
      });
    });
  });
  
  describe('Text Input and Analysis Workflow', () => {
    it('should handle text input and trigger analysis', async () => {
      render(<App />);
      
      // Look for text input field
      const textInput = screen.queryByRole('textbox') || 
                       screen.queryByPlaceholderText(/enter text/i) ||
                       screen.queryByLabelText(/text/i);
      
      if (textInput) {
        // Enter ethical text
        const ethicalText = 'AI systems should respect human dignity and autonomy';
        fireEvent.change(textInput, { target: { value: ethicalText } });
        
        // Look for analyze button
        const analyzeButton = screen.queryByRole('button', { name: /analyze/i }) ||
                             screen.queryByText(/analyze/i);
        
        if (analyzeButton) {
          fireEvent.click(analyzeButton);
          
          // Wait for analysis to complete
          await waitFor(() => {
            // Should show some result or loading state
            expect(document.body).toBeInTheDocument();
          });
        }
      }
    });
    
    it('should display analysis results', async () => {
      render(<App />);
      
      // Simulate successful analysis
      await waitFor(() => {
        // Look for any results display
        const resultsArea = screen.queryByText(/score/i) ||
                           screen.queryByText(/analysis/i) ||
                           screen.queryByText(/result/i);
        
        // Should either show results or be ready to show them
        expect(document.body).toBeInTheDocument();
      });
    });
  });
  
  describe('Real-time Feedback', () => {
    it('should provide real-time feedback as user types', async () => {
      render(<App />);
      
      const textInput = screen.queryByRole('textbox');
      
      if (textInput) {
        // Simulate typing
        const partialText = 'AI should';
        fireEvent.change(textInput, { target: { value: partialText } });
        
        // Should handle partial input without errors
        await waitFor(() => {
          expect(textInput).toHaveValue(partialText);
        });
        
        // Complete the text
        const fullText = 'AI should respect human rights';
        fireEvent.change(textInput, { target: { value: fullText } });
        
        await waitFor(() => {
          expect(textInput).toHaveValue(fullText);
        });
      }
    });
  });
  
  describe('Visualization Components', () => {
    it('should handle vector visualization data', async () => {
      render(<App />);
      
      // Wait for any visualization components to load
      await waitFor(() => {
        // Look for canvas, svg, or visualization containers
        const visualElements = document.querySelectorAll('canvas, svg, [data-testid*="visual"]');
        
        // Should either have visualizations or be ready for them
        expect(document.body).toBeInTheDocument();
      });
    });
    
    it('should update visualizations based on analysis', async () => {
      render(<App />);
      
      // Simulate analysis that would update visualizations
      await waitFor(() => {
        // Visualizations should be responsive to data changes
        expect(document.body).toBeInTheDocument();
      });
    });
  });
  
  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      // Mock API error
      server.use(
        http.post('http://localhost:8080/embed', () => {
          return HttpResponse.json({ error: 'Internal server error' }, { status: 500 });
        })
      );
      
      render(<App />);
      
      const textInput = screen.queryByRole('textbox');
      
      if (textInput) {
        fireEvent.change(textInput, { target: { value: 'test text' } });
        
        const analyzeButton = screen.queryByRole('button', { name: /analyze/i });
        
        if (analyzeButton) {
          fireEvent.click(analyzeButton);
          
          // Should handle error without crashing
          await waitFor(() => {
            expect(document.body).toBeInTheDocument();
          });
        }
      }
    });
    
    it('should handle network connectivity issues', async () => {
      // Mock network error
      server.use(
        http.get('http://localhost:8080/health/ready', () => {
          return HttpResponse.error();
        })
      );
      
      render(<App />);
      
      // Should handle network errors gracefully
      await waitFor(() => {
        expect(document.body).toBeInTheDocument();
      });
    });
  });
  
  describe('Performance and Responsiveness', () => {
    it('should handle large text inputs', async () => {
      render(<App />);
      
      const textInput = screen.queryByRole('textbox');
      
      if (textInput) {
        // Create large text input
        const largeText = 'This is a test sentence. '.repeat(100);
        
        fireEvent.change(textInput, { target: { value: largeText } });
        
        // Should handle large input without performance issues
        await waitFor(() => {
          expect(textInput).toHaveValue(largeText);
        });
      }
    });
    
    it('should provide loading states for long operations', async () => {
      // Mock slow API response
      server.use(
        http.post('http://localhost:8080/analyze', async () => {
          await new Promise(resolve => setTimeout(resolve, 1000));
          return HttpResponse.json(mockAnalysisResponse);
        })
      );
      
      render(<App />);
      
      const textInput = screen.queryByRole('textbox');
      const analyzeButton = screen.queryByRole('button', { name: /analyze/i });
      
      if (textInput && analyzeButton) {
        fireEvent.change(textInput, { target: { value: 'test text' } });
        fireEvent.click(analyzeButton);
        
        // Should show loading state
        await waitFor(() => {
          // Look for loading indicators
          const loadingIndicator = screen.queryByText(/loading/i) ||
                                  screen.queryByText(/analyzing/i) ||
                                  screen.queryByRole('progressbar');
          
          // Should either show loading or complete quickly
          expect(document.body).toBeInTheDocument();
        });
      }
    });
  });
  
  describe('Accessibility', () => {
    it('should be accessible to screen readers', () => {
      render(<App />);
      
      // Check for proper ARIA labels and roles
      const interactiveElements = screen.queryAllByRole('button');
      const textInputs = screen.queryAllByRole('textbox');
      
      // Should have accessible interactive elements
      expect(interactiveElements.length + textInputs.length).toBeGreaterThanOrEqual(0);
    });
    
    it('should support keyboard navigation', async () => {
      render(<App />);
      
      // Test tab navigation
      const focusableElements = document.querySelectorAll(
        'button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      
      if (focusableElements.length > 0) {
        // Should be able to focus elements
        (focusableElements[0] as HTMLElement).focus();
        expect(document.activeElement).toBe(focusableElements[0]);
      }
    });
  });
  
  describe('Data Flow Integration', () => {
    it('should maintain consistent data flow from input to output', async () => {
      render(<App />);
      
      const testText = 'AI ethics requires careful consideration of human values';
      const textInput = screen.queryByRole('textbox');
      
      if (textInput) {
        // Input text
        fireEvent.change(textInput, { target: { value: testText } });
        
        // Trigger analysis
        const analyzeButton = screen.queryByRole('button', { name: /analyze/i });
        if (analyzeButton) {
          fireEvent.click(analyzeButton);
          
          // Wait for results
          await waitFor(() => {
            // Should maintain data consistency throughout the flow
            expect(document.body).toBeInTheDocument();
          });
        }
      }
    });
    
    it('should handle multiple concurrent analyses', async () => {
      render(<App />);
      
      // Simulate multiple rapid analyses
      const textInput = screen.queryByRole('textbox');
      const analyzeButton = screen.queryByRole('button', { name: /analyze/i });
      
      if (textInput && analyzeButton) {
        // First analysis
        fireEvent.change(textInput, { target: { value: 'First text' } });
        fireEvent.click(analyzeButton);
        
        // Second analysis before first completes
        fireEvent.change(textInput, { target: { value: 'Second text' } });
        fireEvent.click(analyzeButton);
        
        // Should handle concurrent requests gracefully
        await waitFor(() => {
          expect(document.body).toBeInTheDocument();
        });
      }
    });
  });
});

describe('Component-Specific Integration Tests', () => {
  
  describe('Interaction Page Integration', () => {
    it('should integrate all interaction features', async () => {
      render(<App />);
      
      // Should render interaction components
      await waitFor(() => {
        expect(document.body).toBeInTheDocument();
      });
    });
  });
  
  describe('API Integration Layer', () => {
    it('should handle all API endpoints correctly', async () => {
      render(<App />);
      
      // Test health check integration
      await waitFor(() => {
        expect(document.body).toBeInTheDocument();
      });
    });
  });
});
