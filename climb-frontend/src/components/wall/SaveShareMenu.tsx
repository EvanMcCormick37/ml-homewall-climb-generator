import { useState, useEffect } from "react";
import {
  Link,
  Check,
  Share2,
  Image,
  Loader2,
  Database,
  Save,
  Lock,
  XCircle,
} from "lucide-react";
import { SectionLabel } from "./ui";

// ─── Shared props ────────────────────────────────────────────────────────────

export interface SaveShareMenuProps {
  onCopyLink: () => void;
  onExportImage: () => void;
  onNativeShare: () => void;
  onSaveToDatabase: () => void;
  isExporting: boolean;
  isSaving: boolean;
  linkCopied: boolean;
  hasNativeShare: boolean;
  isSignedIn: boolean;
  hasClimb: boolean;
  saveSuccess: boolean;
  saveError: string | null;
}

// ─── Mobile: popup menu triggered from FAB ───────────────────────────────────

export function SaveShareMenu({
  onCopyLink,
  onExportImage,
  onNativeShare,
  onSaveToDatabase,
  isExporting,
  isSaving,
  linkCopied,
  hasNativeShare,
  isSignedIn,
  hasClimb,
  saveSuccess,
  saveError,
}: SaveShareMenuProps) {
  const [open, setOpen] = useState(false);

  // Keep menu open while saving; auto-close once saved or after showing error
  useEffect(() => {
    if ((saveSuccess || saveError) && open) {
      const t = setTimeout(() => setOpen(false), 1500);
      return () => clearTimeout(t);
    }
  }, [saveSuccess, saveError, open]);

  const menuBtn: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    width: "100%",
    padding: "10px 14px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.6rem",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    background: "transparent",
    border: "none",
    borderBottom: "1px solid var(--border)",
    color: "var(--text-muted)",
    transition: "all 0.15s",
    textAlign: "left" as const,
  };

  if (!hasClimb) return null;

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          pointerEvents: "auto",
          display: "flex",
          alignItems: "center",
          gap: "7px",
          padding: "10px 18px",
          fontFamily: "'Space Mono', monospace",
          fontSize: "0.65rem",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          background: "var(--cyan)",
          border: "1px solid var(--cyan)",
          color: "#09090b",
          fontWeight: 700,
          cursor: "pointer",
          borderRadius: "var(--radius)",
          boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
        }}
      >
        <Save size={12} /> Save / Share
      </button>

      {open && (
        <>
          <div
            style={{ position: "fixed", inset: 0, zIndex: 50 }}
            onClick={() => setOpen(false)}
          />
          <div
            style={{
              position: "absolute",
              bottom: "calc(100% + 8px)",
              left: "50%",
              transform: "translateX(-50%)",
              minWidth: "220px",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              boxShadow: "0 16px 48px rgba(0,0,0,0.6)",
              zIndex: 60,
              animation: "bzFadeUp 0.15s ease-out",
              overflow: "hidden",
            }}
          >
            <style>{`
              .ssm-btn:hover, .ssm-btn:focus { color: var(--cyan) !important; outline: none; }
            `}</style>
            {/* Copy Link */}
            <button
              onClick={() => {
                onCopyLink();
              }}
              style={menuBtn}
              className="ssm-btn"
            >
              {linkCopied ? (
                <>
                  <Check size={12} style={{ color: "var(--cyan)" }} />
                  <span style={{ color: "var(--cyan)" }}>Link Copied!</span>
                </>
              ) : (
                <>
                  <Link size={12} /> Copy Link
                </>
              )}
            </button>

            {/* Save Image */}
            <button
              onClick={() => {
                onExportImage();
              }}
              disabled={isExporting}
              style={{ ...menuBtn, opacity: isExporting ? 0.5 : 1 }}
              className="ssm-btn"
            >
              {isExporting ? (
                <>
                  <Loader2
                    size={12}
                    style={{ animation: "spin 1s linear infinite" }}
                  />{" "}
                  Rendering…
                </>
              ) : (
                <>
                  <Image size={12} /> Save Image
                </>
              )}
            </button>

            {/* Native Share */}
            {hasNativeShare && (
              <button
                onClick={() => {
                  onNativeShare();
                  setOpen(false);
                }}
                style={menuBtn}
                className="ssm-btn"
              >
                <Share2 size={12} /> Share…
              </button>
            )}

            {/* Save to Database */}
            <button
              onClick={() => {
                if (isSignedIn && !isSaving) onSaveToDatabase();
              }}
              disabled={!isSignedIn || isSaving}
              style={{
                ...menuBtn,
                borderBottom: "none",
                color: !isSignedIn
                  ? "var(--text-dim)"
                  : saveError
                    ? "#f87171"
                    : saveSuccess
                      ? "var(--cyan)"
                      : "var(--text-muted)",
                cursor: !isSignedIn || isSaving ? "not-allowed" : "pointer",
                opacity: !isSignedIn ? 0.5 : 1,
                borderTop: saveError
                  ? "1px solid rgba(248,113,113,0.4)"
                  : saveSuccess
                    ? "1px solid var(--cyan)"
                    : undefined,
                background: saveError
                  ? "rgba(248,113,113,0.1)"
                  : saveSuccess
                    ? "var(--cyan-dim)"
                    : "transparent",
              }}
              className="ssm-btn"
              title={!isSignedIn ? "Sign in to save climbs" : undefined}
            >
              {!isSignedIn ? (
                <>
                  <Lock size={12} /> Set to Database
                  <span
                    style={{
                      fontSize: "0.5rem",
                      color: "var(--text-dim)",
                      marginLeft: "auto",
                    }}
                  >
                    Sign in
                  </span>
                </>
              ) : isSaving ? (
                <>
                  <Loader2
                    size={12}
                    style={{ animation: "spin 1s linear infinite" }}
                  />{" "}
                  Saving…
                </>
              ) : saveError ? (
                <>
                  <XCircle size={12} style={{ color: "#f87171" }} /> {saveError}
                </>
              ) : saveSuccess ? (
                <>
                  <Check size={12} /> Saved!
                </>
              ) : (
                <>
                  <Database size={12} /> Set to Database
                </>
              )}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// ─── Desktop: inline panel for sidebar ───────────────────────────────────────

export function DesktopSaveSharePanel({
  onCopyLink,
  onExportImage,
  onNativeShare,
  onSaveToDatabase,
  isExporting,
  isSaving,
  linkCopied,
  hasNativeShare,
  isSignedIn,
  hasClimb,
  saveSuccess,
  saveError,
}: SaveShareMenuProps) {
  if (!hasClimb) return null;

  const actionBtn: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "7px",
    width: "100%",
    padding: "9px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.6rem",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    background: "transparent",
    border: "1px solid var(--border)",
    color: "var(--text-muted)",
    cursor: "pointer",
    borderRadius: "var(--radius)",
    transition: "all 0.15s",
  };

  return (
    <div
      style={{
        padding: "16px",
        borderTop: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <div
          style={{ width: "2px", height: "10px", background: "var(--cyan)" }}
        />
        <SectionLabel>Save & Share</SectionLabel>
      </div>

      <button onClick={onCopyLink} style={actionBtn}>
        {linkCopied ? (
          <>
            <Check size={10} style={{ color: "var(--cyan)" }} />
            <span style={{ color: "var(--cyan)" }}>Link Copied!</span>
          </>
        ) : (
          <>
            <Link size={10} /> Copy Link
          </>
        )}
      </button>

      <button
        onClick={onExportImage}
        disabled={isExporting}
        style={{ ...actionBtn, opacity: isExporting ? 0.5 : 1 }}
      >
        {isExporting ? (
          <>
            <Loader2
              size={10}
              style={{ animation: "spin 1s linear infinite" }}
            />{" "}
            Rendering…
          </>
        ) : (
          <>
            <Image size={10} /> Save Image
          </>
        )}
      </button>

      {hasNativeShare && (
        <button
          onClick={onNativeShare}
          style={{
            ...actionBtn,
            background: "var(--cyan)",
            color: "#09090b",
            border: "1px solid var(--cyan)",
            fontWeight: 700,
          }}
        >
          <Share2 size={10} /> Share…
        </button>
      )}

      {/* Save to Database */}
      <button
        onClick={() => {
          if (isSignedIn && !isSaving) onSaveToDatabase();
        }}
        disabled={!isSignedIn || isSaving}
        style={{
          ...actionBtn,
          color: !isSignedIn
            ? "var(--text-dim)"
            : saveError
              ? "#f87171"
              : saveSuccess
                ? "var(--cyan)"
                : "var(--text-muted)",
          cursor: !isSignedIn || isSaving ? "not-allowed" : "pointer",
          opacity: !isSignedIn ? 0.5 : 1,
          borderColor: saveError
            ? "rgba(248,113,113,0.4)"
            : saveSuccess
              ? "var(--cyan)"
              : "var(--border)",
          background: saveError
            ? "rgba(248,113,113,0.1)"
            : saveSuccess
              ? "var(--cyan-dim)"
              : "transparent",
        }}
        title={
          !isSignedIn
            ? "Sign in to save climbs to the global database"
            : undefined
        }
      >
        {!isSignedIn ? (
          <>
            <Lock size={10} /> Set to Database
          </>
        ) : isSaving ? (
          <>
            <Loader2
              size={10}
              style={{ animation: "spin 1s linear infinite" }}
            />{" "}
            Saving…
          </>
        ) : saveError ? (
          <>
            <XCircle size={10} style={{ color: "#f87171" }} /> {saveError}
          </>
        ) : saveSuccess ? (
          <>
            <Check size={10} /> Saved!
          </>
        ) : (
          <>
            <Database size={10} /> Set to Database
          </>
        )}
      </button>
    </div>
  );
}
