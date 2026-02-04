export const errorHandler = (err, req, res, next) => {
  const status = err.statusCode || 500;
  const message = err.message || 'Internal server error';
  req.log?.error({ err }, 'request failed');
  res.status(status).json({ error: message });
  next();
};
