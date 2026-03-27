import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { SignIn } from "@clerk/clerk-react";
import { HoldGridCanvas } from "@/components";
import { TITLE_STYLES } from "@/styles";

export const Route = createFileRoute("/signIn")({
  component: SignInPage,
});

function SignInPage() {
  const navigate = useNavigate();

  return (
    <>
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
        {/* Nav */}
        <nav
          style={{
            display: "flex",
            alignItems: "center",
            padding: "20px 32px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <button
            onClick={() => navigate({ to: "/" })}
            className="bz-hero-title"
            style={{
              fontSize: "1.5rem",
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              lineHeight: 1,
            }}
          >
            Beta<span>Zero</span>
          </button>
        </nav>

        {/* Main */}
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            position: "relative",
            padding: "60px 32px",
          }}
        >
          <HoldGridCanvas />

          {/* Left accent line */}
          <div
            style={{
              position: "absolute",
              left: 0,
              top: "10%",
              width: "2px",
              height: "60%",
              background: "linear-gradient(to bottom, transparent, var(--cyan), transparent)",
              opacity: 0.6,
            }}
            aria-hidden="true"
          />

          <div style={{ position: "relative", zIndex: 1 }}>
            <SignIn
              routing="hash"
              appearance={{
                variables: {
                  colorBackground: "#111113",
                  colorInputBackground: "#18181b",
                  colorText: "#ffffff",
                  colorTextSecondary: "#a1a1aa",
                  colorPrimary: "#06b6d4",
                  colorInputText: "#ffffff",
                  colorNeutral: "#ffffff",
                  fontFamily: "'Space Mono', monospace",
                  borderRadius: "4px",
                  fontSize: "13px",
                },
                elements: {
                  card: {
                    border: "1px solid rgba(255,255,255,0.08)",
                    boxShadow: "0 0 0 1px rgba(6,182,212,0.08), 0 16px 48px rgba(0,0,0,0.6)",
                  },
                  headerTitle: {
                    fontFamily: "'Oswald', sans-serif",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                    fontSize: "1.5rem",
                  },
                  headerSubtitle: {
                    fontFamily: "'Space Mono', monospace",
                    fontSize: "0.65rem",
                    letterSpacing: "0.06em",
                  },
                  formButtonPrimary: {
                    fontFamily: "'Space Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    fontSize: "0.7rem",
                    backgroundColor: "#06b6d4",
                    color: "#09090b",
                  },
                  footerActionLink: {
                    color: "#06b6d4",
                  },
                  identityPreviewEditButton: {
                    color: "#06b6d4",
                  },
                },
              }}
            />
          </div>
        </div>
      </div>
    </>
  );
}
