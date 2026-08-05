import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    // Pin the data layer to mock mode for tests, regardless of any .env.local
    // a developer has lying around. Without this, a local file pointing at a
    // running backend silently makes the suite assert against live data — the
    // tests then fail for reasons that have nothing to do with the code, or
    // worse, pass while exercising something other than what they claim to.
    env: { VITE_USE_MOCK: "true", VITE_API_BASE_URL: "" },
  },
});
