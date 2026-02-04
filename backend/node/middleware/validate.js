export const requireFields = (fields = []) => (req, res, next) => {
  const missing = fields.filter((field) => req.body?.[field] === undefined);
  if (missing.length) {
    return res.status(400).json({ error: `Missing fields: ${missing.join(', ')}` });
  }
  return next();
};
