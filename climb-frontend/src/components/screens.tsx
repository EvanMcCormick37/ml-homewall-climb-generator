/** Full-screen loading overlay shown while the server is waking up */
export function WakingScreen() {
  return (
    <div
      style={{
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "32px",
        background: "var(--bg)",
        zIndex: 50,
      }}
    >
      {/* Logo */}
      <img
        src="/logo_transparent.svg"
        alt="BetaZero"
        style={{ width: "clamp(80px, 18vw, 140px)", opacity: 0.9 }}
      />

      {/* Animated dots loader */}
      <div style={{ display: "flex", gap: "10px" }} aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: "var(--cyan)",
              animation: `bzPulse 1.2s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>

      {/* Status text */}
      <div style={{ textAlign: "center" }}>
        <p
          className="bz-mono"
          style={{
            fontSize: "0.75rem",
            color: "var(--text-muted)",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            marginBottom: "6px",
          }}
        >
          The slumbering server is waking up…
        </p>
        <p
          className="bz-mono"
          style={{
            fontSize: "0.65rem",
            color: "rgba(113,113,122,0.55)",
            letterSpacing: "0.06em",
          }}
        >
          This takes about 10-20 seconds on first visit.
        </p>
      </div>

      <style>{`
        @keyframes bzPulse {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.3; }
          40%            { transform: scale(1);   opacity: 1;   }
        }
      `}</style>
    </div>
  );
}
