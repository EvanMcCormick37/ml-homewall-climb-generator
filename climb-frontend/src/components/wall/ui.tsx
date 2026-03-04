import type React from "react";

/** A section label consistent with homepage eyebrow style */
export function SectionLabel({
  desc,
  children,
}: {
  desc?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <span
        className="bz-mono"
        style={{
          fontSize: "0.6rem",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
        }}
      >
        {children}
      </span>
      <p
        className="bz-mono"
        style={{
          fontSize: "0.6rem",
          color: "var(--text-muted)",
          lineHeight: 1.7,
          paddingTop: "10px",
          borderTop: "1px solid var(--border)",
        }}
      >
        {desc}
      </p>
    </div>
  );
}

/** Toggle button pair (e.g. By Role / Uniform) */
export function TogglePair({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "flex", gap: "2px", width: "100%" }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            flex: value == opt.value ? "1 0 auto" : "1 1 30%",
            padding: "6px 8px",
            minWidth: 0,
            fontSize: "clamp(0.35rem, 2.5vw, 0.65rem)",
            fontFamily: "'Space Mono', monospace",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            overflow: "hidden",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            border: `1px solid ${value === opt.value ? "var(--cyan)" : "var(--border)"}`,
            background: value === opt.value ? "var(--cyan-dim)" : "transparent",
            color: value === opt.value ? "var(--cyan)" : "var(--text-muted)",
            cursor: "pointer",
            transition: "all 0.15s",
            borderRadius: "var(--radius)",
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/** Labeled range slider */
export function BzRange({
  label,
  desc,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
  leftLabel,
  rightLabel,
}: {
  label: string;
  desc?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  displayValue: string;
  leftLabel?: string;
  rightLabel?: string;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "8px",
        }}
      >
        <SectionLabel desc={desc}>{label}</SectionLabel>
        <span
          className="bz-mono"
          style={{ fontSize: "0.65rem", color: "var(--cyan)" }}
        >
          {displayValue}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="bz-range"
      />
      {(leftLabel || rightLabel) && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "4px",
          }}
        >
          <span
            className="bz-mono"
            style={{ fontSize: "0.55rem", color: "var(--text-dim)" }}
          >
            {leftLabel}
          </span>
          <span
            className="bz-mono"
            style={{ fontSize: "0.55rem", color: "var(--text-dim)" }}
          >
            {rightLabel}
          </span>
        </div>
      )}
    </div>
  );
}
