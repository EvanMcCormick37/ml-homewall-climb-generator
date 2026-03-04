import { SignedIn, SignedOut, UserButton } from "@clerk/clerk-react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useWalls } from "@/hooks/useWalls";
import { getWallPhotoUrl } from "@/api/walls";
import { WakingScreen } from "@/components";
import { useEffect, useRef } from "react";
import { TITLE_STYLES } from "@/styles";

export const Route = createFileRoute("/")({
  component: HomePage,
});

const LINKS = [
  { label: "About Me", href: "https://www.evmojo.dev" },
  {
    label: "Github",
    href: "https://github.com/EvanMcCormick37/ml-homewall-climb-generator",
  },
  {
    label: "LSTM Model (v1)",
    href: "https://evmojo37.substack.com/p/beta-zero-alpha-can-ai-set-climbs",
  },
  {
    label: "DDPM Model (v2)",
    href: "https://evmojo37.substack.com/p/betazero-v2-a-diffusion-model-for",
  },
  {
    label: "This App",
    href: "https://open.substack.com/pub/evmojo37/p/how-to-turn-an-ml-model-into-a-functioning",
  },
  {
    label: "Buy me a Coffee",
    href: "https://ko-fi.com/B0B81E9CGS",
  },
  {
    label: "Give Feedback",
    href: "https://docs.google.com/forms/d/e/1FAIpQLSeYDIel5MMjj0X3zlXFe4N4FZdUcXadAL5bR-Wjb4W7SVZ5SQ/viewform?usp=publish-editor",
  },
];

// Generates a subtle hold-grid dot pattern on a canvas
function HoldGridCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      draw();
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const spacing = 48;
      const cols = Math.ceil(canvas.width / spacing) + 1;
      const rows = Math.ceil(canvas.height / spacing) + 1;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const jitter = spacing * 0.18;
          const x = c * spacing + (Math.random() - 0.5) * jitter;
          const y = r * spacing + (Math.random() - 0.5) * jitter;

          const opacity = 0.04 + Math.random() * 0.09;
          const radius = 2 + Math.random() * 3;

          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(6, 182, 212, ${opacity})`;
          ctx.fill();
        }
      }
    };

    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    />
  );
}

function HomePage() {
  const { walls, loading, waking, error } = useWalls();
  const navigate = useNavigate();

  return (
    <>
      {/* ── Google Font: Space Mono (monospace, technical feel) + Oswald (condensed display) ── */}
      <style>{TITLE_STYLES}</style>

      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          background: "var(--bg)",
          color: "var(--text-primary)",
        }}
      >
        {/* ── Top Nav ── */}
        <nav
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "20px 32px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: "28px",
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            {LINKS.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="bz-nav-link"
              >
                {link.label}
              </a>
            ))}
          </div>
          {/* Desktop Sign-In */}
          <div className="hidden lg:flex">
            <SignedOut>
              <button
                onClick={() => navigate({ to: "/signIn" })}
                className="bz-mono"
                style={{
                  fontSize: "0.65rem",
                  color: "var(--cyan)",
                  background: "var(--surface)",
                  border: "1px solid var(--cyan)",
                  borderRadius: "4px",
                  padding: "4px 12px",
                  cursor: "pointer",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                }}
              >
                Sign In
              </button>
            </SignedOut>
          </div>
          <SignedIn>
            <UserButton
              appearance={{
                elements: {
                  avatarBox: { width: 32, height: 32 },
                },
              }}
            />
          </SignedIn>
        </nav>

        {/* {Mobile Sign-in} */}
        <div
          style={{
            position: "fixed",
            bottom: "48px",
            left: 0,
            right: 0,
            justifyContent: "center",
            gap: "10px",
            zIndex: 30,
            padding: "0 16px",
          }}
          className="flex lg:hidden"
        >
          <SignedOut>
            <button
              onClick={() => navigate({ to: "/signIn" })}
              className="bz-mono"
              style={{
                fontSize: "0.65rem",
                color: "var(--cyan)",
                background: "var(--surface)",
                border: "1px solid var(--cyan)",
                borderRadius: "4px",
                padding: "4px 12px",
                cursor: "pointer",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              Sign In
            </button>
          </SignedOut>
        </div>

        {/* ── Hero ── */}
        <section
          style={{
            position: "relative",
            padding: "clamp(48px, 10vw, 96px) 32px clamp(56px, 10vw, 80px)",
            overflow: "hidden",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <HoldGridCanvas />

          <div
            style={{
              position: "absolute",
              left: 0,
              top: "10%",
              width: "2px",
              height: "60%",
              background:
                "linear-gradient(to bottom, transparent, var(--cyan), transparent)",
              opacity: 0.6,
            }}
            aria-hidden="true"
          />

          <div
            style={{
              position: "relative",
              maxWidth: "1100px",
              margin: "0 auto",
            }}
          >
            <div
              className="bz-anim bz-anim-1"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                marginBottom: "24px",
              }}
            >
              <div className="bz-accent-bar" />
              <span className="bz-section-label">Welcome to</span>
            </div>

            <h1 className="bz-hero-title bz-anim bz-anim-2">
              Beta<span>Zero</span>
            </h1>

            <p
              className="bz-mono bz-anim bz-anim-3"
              style={{
                marginTop: "32px",
                maxWidth: "480px",
                fontSize: "clamp(0.8rem, 1.5vw, 0.95rem)",
                lineHeight: 1.75,
                color: "var(--text-muted)",
              }}
            >
              A public resource for generating board climbs using machine
              learning.
            </p>

            <div
              className="bz-anim bz-anim-4"
              style={{
                marginTop: "56px",
                display: "flex",
                alignItems: "center",
                gap: "10px",
              }}
            >
              <span className="bz-section-label">Choose your wall</span>
              <svg
                className="bz-scroll-cue"
                width="14"
                height="14"
                viewBox="0 0 14 14"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M7 2v10M2 8l5 4 5-4"
                  stroke="var(--cyan)"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          </div>
        </section>

        {/* ── Wall Selection ── */}
        <section
          style={{
            flex: 1,
            padding: "clamp(40px, 8vw, 72px) 32px",
            maxWidth: "1100px",
            margin: "0 auto",
            width: "100%",
          }}
        >
          <div className="bz-anim bz-anim-5">
            {/* Initial loading (first attempt, not yet known if 502) */}
            {loading && (
              <div
                className="bz-mono"
                style={{
                  color: "var(--text-muted)",
                  fontSize: "0.8rem",
                  padding: "60px 0",
                  textAlign: "center",
                }}
              >
                — loading walls —
              </div>
            )}
            {waking && <WakingScreen />}
            {/* Error */}
            {error && (
              <div
                className="bz-mono"
                style={{
                  fontSize: "0.65rem",
                  color: "#f87171",
                  background: "rgba(248,113,113,0.08)",
                  border: "1px solid rgba(248,113,113,0.2)",
                  borderRadius: "var(--radius)",
                  padding: "8px 10px",
                }}
              >
                {error}
              </div>
            )}

            {/* Cards */}
            {!loading && !error && (
              <div className="bz-wall-grid">
                {/* Add your wall card — signed-in users only */}
                <SignedIn>
                  <button
                    onClick={() => navigate({ to: "/walls/new" })}
                    className="bz-card"
                    style={{
                      position: "relative",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      minHeight: "240px",
                      border: "1px dashed rgba(6,182,212,0.3)",
                      background: "transparent",
                      gap: "12px",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--cyan)";
                      e.currentTarget.style.background = "var(--cyan-dim)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor =
                        "rgba(6,182,212,0.3)";
                      e.currentTarget.style.background = "transparent";
                    }}
                  >
                    <div
                      style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "50%",
                        border: "1px solid var(--cyan)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "var(--cyan)",
                      }}
                    >
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                      </svg>
                    </div>
                    <div
                      style={{
                        fontFamily: "'Oswald', sans-serif",
                        fontSize: "1rem",
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        color: "var(--cyan)",
                      }}
                    >
                      Add Your Wall
                    </div>
                    <div
                      className="bz-mono"
                      style={{
                        fontSize: "0.6rem",
                        color: "var(--text-muted)",
                        letterSpacing: "0.06em",
                        textAlign: "center",
                        padding: "0 16px",
                      }}
                    >
                      Upload a photo and mark your holds
                    </div>
                  </button>
                </SignedIn>

                {walls.map((wall) => (
                  <button
                    key={wall.id}
                    onClick={() =>
                      navigate({
                        to: "/$wallId/set",
                        params: { wallId: wall.id },
                      })
                    }
                    className="bz-card"
                    style={{ position: "relative" }}
                  >
                    {/* Photo */}
                    <div
                      style={{
                        width: "100%",
                        height: "180px",
                        overflow: "hidden",
                        background: "#1c1c1e",
                      }}
                    >
                      <img
                        src={getWallPhotoUrl(wall.id)}
                        alt={wall.name}
                        className="bz-card-img"
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                          display: "block",
                        }}
                      />
                    </div>

                    {/* Info */}
                    <div style={{ padding: "16px 18px 18px" }}>
                      <div
                        style={{
                          fontFamily: "'Oswald', sans-serif",
                          fontSize: "1.15rem",
                          fontWeight: 700,
                          textTransform: "uppercase",
                          letterSpacing: "0.04em",
                          color: "var(--text-primary)",
                          marginBottom: "6px",
                        }}
                      >
                        {wall.name}
                      </div>
                      <div
                        className="bz-mono"
                        style={{
                          fontSize: "0.65rem",
                          color: "var(--text-muted)",
                          letterSpacing: "0.06em",
                        }}
                      >
                        {wall.num_holds} holds
                        {wall.dimensions &&
                          ` · ${wall.dimensions[0]}×${wall.dimensions[1]} ft`}
                        {wall.angle != null && ` · ${wall.angle}°`}
                      </div>
                    </div>

                    {/* Bottom accent line */}
                    <div
                      style={{
                        height: "2px",
                        background: "var(--cyan)",
                        opacity: 0.9,
                      }}
                    />
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* ── Footer ── */}
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "20px 32px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          <span
            className="bz-mono"
            style={{
              fontSize: "0.65rem",
              color: "var(--text-muted)",
              letterSpacing: "0.1em",
            }}
          >
            {new Date().getFullYear()} Evan McCormick
          </span>
          <span
            className="bz-mono"
            style={{
              fontSize: "0.65rem",
              color: "var(--text-muted)",
              letterSpacing: "0.1em",
            }}
          >
            MIT License
          </span>
        </footer>
      </div>
    </>
  );
}
