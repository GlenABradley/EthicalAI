import React from 'react';
import { render, screen } from '../test-utils';
import DecisionProof from '../../components/DecisionProof';

describe('DecisionProof', () => {
  const mockProof = {
    spans: [],
    final: {
      action: 'allow',
      rationale: 'This action is allowed'
    }
  };

  it('renders the decision', () => {
    const { container } = render(<DecisionProof proof={mockProof} />);
    const actionDiv = container.querySelector('div[style*="font-size: 14px"]');
    expect(actionDiv).toBeTruthy();
    expect(actionDiv?.textContent).toContain('Action:');
    expect(actionDiv?.textContent).toContain('allow');
  });

  it('displays rationale when provided', () => {
    const { container } = render(<DecisionProof proof={mockProof} />);
    expect(container.textContent).toContain(mockProof.final.rationale);
  });

  it('handles spans when present', () => {
    const withSpans = {
      final: { action: 'allow', rationale: 'Test with spans' },
      spans: [
        { i: 0, j: 5, axis: 'safety', score: 0.8, threshold: 0.7 },
      ],
    };
    const { container } = render(<DecisionProof proof={withSpans} />);
    const table = container.querySelector('table');
    expect(table).toBeTruthy();
    expect(container.textContent).toContain('0-5');
    expect(container.textContent).toContain('safety');
    expect(container.textContent).toContain('0.800');
    expect(container.textContent).toContain('0.700');
  });

  it('handles spans when present', () => {
    const withSpans = {
      final: { action: 'allow', rationale: 'Test with spans' },
      spans: [
        { i: 0, j: 5, axis: 'safety', score: 0.8, threshold: 0.7 },
      ],
    };
    render(<DecisionProof proof={withSpans} />);
    expect(screen.getByText('0-5')).toBeInTheDocument();
    expect(screen.getByText('safety')).toBeInTheDocument();
    expect(screen.getByText('0.800')).toBeInTheDocument();
    expect(screen.getByText('0.700')).toBeInTheDocument();
  });
});
