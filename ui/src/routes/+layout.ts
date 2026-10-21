// Static SPA (vite.config.ts adapter-static + fallback): every route is
// client-rendered, so a `load` function never runs against a build-time
// backend that doesn't exist.
export const ssr = false;
