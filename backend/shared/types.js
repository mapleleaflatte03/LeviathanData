/**
 * @typedef {Object} Envelope
 * @property {string} type
 * @property {string} requestId
 * @property {string} ts
 * @property {any} payload
 */

/**
 * @typedef {Object} PipelineStatus
 * @property {string} runId
 * @property {string} stage
 * @property {string} status
 * @property {string} message
 */

export const noop = () => {};
