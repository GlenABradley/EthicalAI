// Import testing library matchers
import '@testing-library/jest-dom/vitest';
import { render, RenderOptions } from '@testing-library/react';
import { ReactElement, ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';

interface AllTheProvidersProps {
  children: ReactNode;
}

const AllTheProviders = ({ children }: AllTheProvidersProps) => {
  return (
    <BrowserRouter>
      {children}
    </BrowserRouter>
  );
};

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {}

const customRender = (
  ui: ReactElement,
  options?: CustomRenderOptions,
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything
export * from '@testing-library/react';
// Override render method
export { customRender as render };

// Export userEvent for convenience
export { default as userEvent } from '@testing-library/user-event';
