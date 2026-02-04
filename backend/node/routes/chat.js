import { Router } from 'express';
import { chatCompletionText } from '../llmClient.js';

const router = Router();

router.post('/', async (req, res) => {
  const messages = req.body?.messages || [];
  try {
    const text = await chatCompletionText(messages);
    return res.json({ text });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

export default router;
