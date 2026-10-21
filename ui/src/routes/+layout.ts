// Static SPA (adapter-static + fallback in vite.config.ts): every route is
// client-rendered, so a `load` function never runs against a build-time
// backend that doesn't exist.
export const ssr = false;
