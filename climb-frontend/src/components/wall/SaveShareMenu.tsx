import { useState } from "react";
import {
  Link,
  Check,
  Share2,
  Image,
  Loader2,
  Database,
  Save,
  Lock,
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
}: SaveShareMenuProps) {
  const [open, setOpen] = useState(false);

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
    cursor: "pointer",
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
            {/* Copy Link */}
            <button onClick={() => { onCopyLink(); }} style={menuBtn}>
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
              onClick={() => { onExportImage(); }}
              disabled={isExporting}
              style={{ ...menuBtn, opacity: isExporting ? 0.5 : 1 }}
            >
              {isExporting ? (
                <>
                  <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> Rendering…
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
                onClick={() => { onNativeShare(); setOpen(false); }}
                style={menuBtn}
              >
                <Share2 size={12} /> Share…
              </button>
            )}

            {/* Save to Database */}
            <button
              onClick={() => {
                if (isSignedIn && !isSaving) {
                  onSaveToDatabase();
                  setOpen(false);
                }
              }}
              disabled={!isSignedIn || isSaving}
              style={{
                ...menuBtn,
                borderBottom: "none",
                color: !isSignedIn
                  ? "var(--text-dim)"
                  : saveSuccess
                    ? "var(--cyan)"
                    : "var(--text-muted)",
                cursor: !isSignedIn || isSaving ? "not-allowed" : "pointer",
                opacity: !isSignedIn ? 0.5 : 1,
              }}
              title={!isSignedIn ? "Sign in to save climbs" : undefined}
            >
              {!isSignedIn ? (
                <>
                  <Lock size={12} /> Set to Database
                  <span style={{ fontSize: "0.5rem", color: "var(--text-dim)", marginLeft: "auto" }}>
                    Sign in
                  </span>
                </>
              ) : isSaving ? (
                <>
                  <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> Saving…
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
        <div style={{ width: "2px", height: "10px", background: "var(--cyan)" }} />
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
            <Loader2 size={10} style={{ animation: "spin 1s linear infinite" }} /> Rendering…
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
            : saveSuccess
              ? "var(--cyan)"
              : "var(--text-muted)",
          cursor: !isSignedIn || isSaving ? "not-allowed" : "pointer",
          opacity: !isSignedIn ? 0.5 : 1,
          borderColor: saveSuccess ? "var(--cyan)" : "var(--border)",
          background: saveSuccess ? "var(--cyan-dim)" : "transparent",
        }}
        title={!isSignedIn ? "Sign in to save climbs to the global database" : undefined}
      >
        {!isSignedIn ? (
          <>
            <Lock size={10} /> Set to Database
          </>
        ) : isSaving ? (
          <>
            <Loader2 size={10} style={{ animation: "spin 1s linear infinite" }} /> Saving…
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
