/**
 * Playwright UI smoke for the local catalog.
 * The ten named cases are implemented in tests/test_local_catalog.py
 * (stub llmfit + mock Ollama; no network except optional AGENTFORGE_LIVE_RAG=1).
 */
import { test, expect } from "@playwright/test";

test("catalog table labels estimates and keeps OpenRouter separate", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("text=OpenRouter")).toBeVisible();
  await expect(page.locator("text=est. VRAM")).toBeVisible();
  await expect(page.locator("text=estimates")).toBeVisible();
  await expect(page.locator("#catList")).toBeVisible();
});
