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
    --text-muted: #d2d2d2;
    --text-dim: #a7a7a7;
    --radius: 4px;
  }
  
  /* Scroll-bars */
  /* 1. Firefox Styling */
  :root {
    /* 'thin' or 'auto' */
    scrollbar-width: thin; 
    /* First color is the "thumb" (the part you drag), second is the "track" (the background) */
    scrollbar-color: var(--cyan-dim) var(--bg); 
  }

  /* 2. WebKit Styling (Chrome, Safari, Edge) */
  /* Targets the overall scrollbar */
  :root::-webkit-scrollbar {
    width: 8px; /* Makes the scrollbar slimmer than the default */
  }

  /* Targets the background track */
  :root::-webkit-scrollbar-track {
    background: var(--bg); /* A dark color to blend with your panel */
    border-radius: 4px;
  }

  /* Targets the draggable handle */
  :root::-webkit-scrollbar-thumb {
    background: var(--cyan-dim); /* A lighter gray so it's visible */
    border-radius: 4px;
  }

  /* Optional: Add a hover effect to the handle */
  :root::-webkit-scrollbar-thumb:hover {
    background: var(--cyan); /* Matches the cyan/teal accent from your UI! */
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

export const TITLE_STYLES = `
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&family=Space+Mono:wght@400;700&display=swap');

        :root {
          --cyan: #06b6d4;
          --cyan-dim: rgba(6,182,212,0.15);
          --bg: #09090b;
          --surface: #111113;
          --border: rgba(255,255,255,0.08);
          --text-primary: #f4f4f5;
          --text-muted: #c7c7c7;
        }

        * { box-sizing: border-box; }

        body { background: var(--bg); }

        .bz-hero-title {
          font-family: 'Oswald', sans-serif;
          font-size: clamp(4.5rem, 14vw, 11rem);
          font-weight: 700;
          line-height: 0.9;
          letter-spacing: -0.02em;
          color: var(--text-primary);
          text-transform: uppercase;
        }

        .bz-hero-title span {
          color: var(--cyan);
        }

        .bz-mono {
          font-family: 'Space Mono', monospace;
        }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .bz-anim { opacity: 0; animation: fadeUp 0.7s ease forwards; }
        .bz-anim-1 { animation-delay: 0.05s; }
        .bz-anim-2 { animation-delay: 0.2s; }
        .bz-anim-3 { animation-delay: 0.35s; }
        .bz-anim-4 { animation-delay: 0.5s; }
        .bz-anim-5 { animation-delay: 0.65s; }

        .bz-rule {
          border: none;
          border-top: 1px solid var(--border);
        }

        .bz-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 4px;
          overflow: hidden;
          cursor: pointer;
          transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
          text-align: left;
          width: 100%;
        }
        .bz-card:hover {
          border-color: var(--cyan);
          transform: translateY(-2px);
          box-shadow: 0 8px 32px rgba(6,182,212,0.12);
        }
        .bz-card:hover .bz-card-img {
          transform: scale(1.04);
        }
        .bz-card-img {
          transition: transform 0.35s ease;
        }

        .bz-nav-link {
          font-family: 'Space Mono', monospace;
          font-size: 0.5rem;
          color: var(--text-muted);
          text-decoration: none;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          transition: color 0.15s;
        }
        .bz-nav-link:hover { color: var(--cyan); }

        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(6px); }
        }
        .bz-scroll-cue {
          animation: bounce 1.8s ease-in-out infinite;
        }

        .bz-section-label {
          font-family: 'Space Mono', monospace;
          font-size: 0.65rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-muted);
        }

        .bz-accent-bar {
          width: 40px;
          height: 2px;
          background: var(--cyan);
          flex-shrink: 0;
        }

        .bz-wall-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
          gap: 16px;
        }
      `;
