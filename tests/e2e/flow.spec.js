import { test, expect } from '@playwright/test';

const BASE_URL = process.env.TEST_URL || 'http://localhost:3000';
const uniqueEmail = () => `user_${Date.now()}@example.com`;

// ─────────────────────────────────────────────────────────────
// Static UI Tests (no backend required)
// ─────────────────────────────────────────────────────────────

test.describe('Static UI', () => {
  test('login screen loads with Leviathan branding', async ({ page }) => {
    await page.goto(BASE_URL);
    await expect(page.getByText('Leviathan')).toBeVisible();
    await expect(page.getByText('Enter the Abyss')).toBeVisible();
  });

  test('toggle between login and register forms', async ({ page }) => {
    await page.goto(BASE_URL);
    await expect(page.locator('#loginForm')).toBeVisible();
    await expect(page.locator('#registerForm')).not.toBeVisible();
    
    await page.getByText('Register').click();
    await expect(page.locator('#registerForm')).toBeVisible();
    await expect(page.locator('#loginForm')).not.toBeVisible();
    
    await page.getByText('Sign In').click();
    await expect(page.locator('#loginForm')).toBeVisible();
  });

  test('theme colors are dark ocean (#001122)', async ({ page }) => {
    await page.goto(BASE_URL);
    const body = page.locator('body');
    const bg = await body.evaluate(el => getComputedStyle(el).backgroundColor);
    expect(bg).toMatch(/rgb\(0, 17, 34\)|#001122/i);
  });
});

// ─────────────────────────────────────────────────────────────
// Full Flow Tests (require backend services)
// ─────────────────────────────────────────────────────────────

test.describe('Full Flow (backend required)', () => {
  test('register new user and reach dashboard', async ({ page }) => {
    const email = uniqueEmail();
    await page.goto(BASE_URL);
    
    await page.getByText('Register').click();
    await page.locator('#registerForm input[name="email"]').fill(email);
    await page.locator('#registerForm input[name="password"]').fill('TestPass123!');
    await page.locator('#registerForm button[type="submit"]').click();
    
    await expect(page.getByText('Swarm Console')).toBeVisible({ timeout: 10_000 });
  });

  test('login with existing user', async ({ page }) => {
    const email = uniqueEmail();
    await page.goto(BASE_URL);
    
    // First register
    await page.getByText('Register').click();
    await page.locator('#registerForm input[name="email"]').fill(email);
    await page.locator('#registerForm input[name="password"]').fill('TestPass123!');
    await page.locator('#registerForm button[type="submit"]').click();
    await expect(page.getByText('Swarm Console')).toBeVisible({ timeout: 10_000 });
    
    // Logout
    await page.locator('#logoutBtn').click();
    await expect(page.getByText('Enter the Abyss')).toBeVisible();
    
    // Login again
    await page.locator('#loginForm input[name="email"]').fill(email);
    await page.locator('#loginForm input[name="password"]').fill('TestPass123!');
    await page.locator('#loginForm button[type="submit"]').click();
    await expect(page.getByText('Swarm Console')).toBeVisible({ timeout: 10_000 });
  });

  test('send chat message and receive response', async ({ page }) => {
    const email = uniqueEmail();
    await page.goto(BASE_URL);
    
    // Register
    await page.getByText('Register').click();
    await page.locator('#registerForm input[name="email"]').fill(email);
    await page.locator('#registerForm input[name="password"]').fill('TestPass123!');
    await page.locator('#registerForm button[type="submit"]').click();
    await expect(page.getByText('Swarm Console')).toBeVisible({ timeout: 10_000 });
    
    // Send message
    await page.locator('#chatInput').fill('Hello Leviathan');
    await page.locator('#sendBtn').click();
    
    // Should see user message
    await expect(page.locator('.message.user')).toContainText('Hello Leviathan');
    
    // Should see assistant response (may take time)
    await expect(page.locator('.message.assistant')).toBeVisible({ timeout: 30_000 });
  });

  test('health endpoint returns ok', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/health`);
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.status).toBe('ok');
  });

  test('alerts endpoint requires auth', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/alerts`);
    expect(response.status()).toBe(401);
  });
});
