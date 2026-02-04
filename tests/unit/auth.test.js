import request from 'supertest';
import path from 'node:path';

let app;

beforeAll(async () => {
  process.env.DB_PATH = path.resolve(process.cwd(), 'data', `test-${Date.now()}.db`);
  const mod = await import('../../backend/node/app.js');
  const created = await mod.createApp({ startPython: false, startBackground: false });
  app = created.app;
});

test('register, login, refresh token rotation', async () => {
  const email = `user_${Date.now()}@example.com`;
  const password = 'TestPass123!';

  const registerRes = await request(app)
    .post('/api/auth/register')
    .send({ email, password });
  expect(registerRes.statusCode).toBe(200);
  expect(registerRes.body.accessToken).toBeTruthy();
  expect(registerRes.body.refreshToken).toBeTruthy();
  const userId = registerRes.body.user.id;
  const refreshToken = registerRes.body.refreshToken;

  const loginRes = await request(app)
    .post('/api/auth/login')
    .send({ email, password });
  expect(loginRes.statusCode).toBe(200);
  expect(loginRes.body.accessToken).toBeTruthy();

  const refreshRes = await request(app)
    .post('/api/auth/refresh')
    .send({ userId, refreshToken });
  expect(refreshRes.statusCode).toBe(200);
  expect(refreshRes.body.accessToken).toBeTruthy();

  const refreshAgain = await request(app)
    .post('/api/auth/refresh')
    .send({ userId, refreshToken });
  expect(refreshAgain.statusCode).toBe(401);
});

test('protected route requires auth', async () => {
  const res = await request(app).get('/api/alerts');
  expect(res.statusCode).toBe(401);
});
