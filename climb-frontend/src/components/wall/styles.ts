/**
 * Shared CSS styles for wall pages (set + view).
 * Inject via <style>{GLOBAL_STYLES}</style> at the top of each page.
 */

export const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&family=Space+Mono:wght@400;700&display=swap');

  :root {
    --cyan: #06b6d4;
    --cyan-dim: rgba(6,182,212,0.15);
    --cyan-glow: rgba(6,182,212,0.25);
    --ruby: #5a0e15;
    --bg: #09090b;
    --surface: #111113;
    --surface2: #18181b;
    --border: rgba(255,255,255,0.08);
    --border-active: #06b6d4;
    --text-primary: #f4f4f5;
    --text-muted: #71717a;
    --text-dim: #3f3f46;
    --radius: 4px;
  }

  /* Typography helpers */
  .bz-oswald {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .bz-mono {
    font-family: 'Space Mono', monospace;
  }

  /* Range input reset */
  .bz-range {
    -webkit-appearance: none;
    width: 100%;
    height: 2px;
    background: rgba(255,255,255,0.1);
    border-radius: 0;
    cursor: pointer;
  }
  .bz-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    background: var(--cyan);
    border-radius: 0;
    cursor: pointer;
  }
  .bz-range::-moz-range-thumb {
    width: 12px;
    height: 12px;
    background: var(--cyan);
    border-radius: 0;
    border: none;
    cursor: pointer;
  }

  /* Slide-in animations for mobile drawers */
  @keyframes bzSlideInLeft {
    from { transform: translateX(-100%); }
    to   { transform: translateX(0); }
  }
  @keyframes bzSlideInRight {
    from { transform: translateX(100%); }
    to   { transform: translateX(0); }
  }
  @keyframes bzFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes bzFadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  select option { background: #111113; color: #f4f4f5; }
`;
