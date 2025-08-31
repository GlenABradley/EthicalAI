import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../App';

describe('App', () => {
  it('renders the app with header and main content', () => {
    const { container } = render(<App />);
    
    // Check for the main header
    const header = container.querySelector('header');
    expect(header).toBeTruthy();
    
    // Check for the main content area
    const main = container.querySelector('main');
    expect(main).toBeTruthy();
  });

  it('renders navigation buttons', () => {
    const { container } = render(<App />);
    
    // Check for navigation buttons
    const buttons = container.querySelectorAll('button');
    const buttonTexts = Array.from(buttons).map(button => button.textContent || '');
    
    expect(buttonTexts).toContain('status');
    expect(buttonTexts).toContain('axes');
    expect(buttonTexts).toContain('analyze');
    expect(buttonTexts).toContain('interaction');
  });
});
