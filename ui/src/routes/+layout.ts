// Static SPA (svelte.config.js adapter-static + fallback): every route is
// client-rendered, so a `load` function never runs against a build-time
// backend that doesn't exist.
export const ssr = false;
