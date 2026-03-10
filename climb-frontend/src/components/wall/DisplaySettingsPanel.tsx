import {
  CATEGORY_ORDER,
  CATEGORY_LABELS,
  type HoldCategory,
  type ColorMode,
  type DisplaySettings,
} from "./types";
import { SectionLabel, TogglePair, BzRange } from "./ui";

export function DisplaySettingsPanel({
  settings,
  onChange,
}: {
  settings: DisplaySettings;
  onChange: (s: DisplaySettings) => void;
}) {
  const update = (patch: Partial<DisplaySettings>) =>
    onChange({ ...settings, ...patch });
  const updateCategoryColor = (cat: HoldCategory, color: string) =>
    update({ categoryColors: { ...settings.categoryColors, [cat]: color } });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        minWidth: "240px",
      }}
    >
      <BzRange
        label="Hold Scale"
        value={settings.scale}
        min={0.3}
        max={3.0}
        step={0.1}
        onChange={(v) => update({ scale: v })}
        displayValue={`${settings.scale.toFixed(1)}×`}
      />

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel>Color Mode</SectionLabel>
        </div>
        <TogglePair
          options={[
            { value: "role", label: "By Role" },
            { value: "uniform", label: "Uniform" },
          ]}
          value={settings.colorMode}
          onChange={(v) => update({ colorMode: v as ColorMode })}
        />
        {settings.colorMode === "uniform" ? (
          <div
            style={{
              marginTop: "10px",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <input
              type="color"
              value={settings.uniformColor}
              onChange={(e) => update({ uniformColor: e.target.value })}
              style={{
                width: "28px",
                height: "28px",
                border: "1px solid var(--border)",
                background: "transparent",
                cursor: "pointer",
                borderRadius: "var(--radius)",
              }}
            />
            <span
              className="bz-mono"
              style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}
            >
              {settings.uniformColor}
            </span>
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "6px",
              marginTop: "10px",
            }}
          >
            {CATEGORY_ORDER.map((cat) => (
              <div
                key={cat}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "6px 8px",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  background: "var(--surface)",
                }}
              >
                <input
                  type="color"
                  value={settings.categoryColors[cat]}
                  onChange={(e) => updateCategoryColor(cat, e.target.value)}
                  style={{
                    width: "18px",
                    height: "18px",
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    padding: 0,
                  }}
                />
                <span
                  className="bz-mono"
                  style={{ fontSize: "0.6rem", color: "var(--text-primary)" }}
                >
                  {CATEGORY_LABELS[cat]}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <BzRange
        label="Opacity"
        value={settings.opacity}
        min={0.1}
        max={1.0}
        step={0.05}
        onChange={(v) => update({ opacity: v })}
        displayValue={`${Math.round(settings.opacity * 100)}%`}
      />

      <div>
        <div style={{ marginBottom: "8px" }}>
          <SectionLabel>Style</SectionLabel>
        </div>
        <TogglePair
          options={[
            { value: "filled", label: "Filled" },
            { value: "outline", label: "Outline" },
          ]}
          value={settings.filled ? "filled" : "outline"}
          onChange={(v) => update({ filled: v === "filled" })}
        />
      </div>
    </div>
  );
}
