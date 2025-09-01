import { describe, it, expect } from "vitest";

describe("env", () => {
  it("has a default API base", () => {
    expect(import.meta.env.VITE_API_BASE ?? "http://localhost:8000").toBeTypeOf("string");
  });
});
