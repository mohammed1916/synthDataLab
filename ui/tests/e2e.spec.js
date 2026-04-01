import { test, expect } from '@playwright/test';

test('homepage has title and create-run form', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'synthDataLab Production Dashboard' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Create Run' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start run' })).toBeVisible();
});

// A simple integration smoke test that uses mocked network response. This does not require backend.
test('run creation shows error when no input is provided', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Start run' }).click();
  await expect(page.getByText('Please provide input text or input path.')).toBeVisible();
});
