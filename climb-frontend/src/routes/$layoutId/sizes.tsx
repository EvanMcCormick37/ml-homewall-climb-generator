import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useRef, useCallback, useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import { getLayout, getLayoutPhotoUrl } from "@/api/layouts";
import { createSize, deleteSize } from "@/api/sizes";
import { ArrowLeft, Trash2 } from "lucide-react";
import { GLOBAL_STYLES } from "@/styles";
import type { LayoutDetail, SizeMetadata } from "@/types";

export const Route = createFileRoute("/$layoutId/sizes")({
  component: SizesPage,
  loader: async ({ params }) => {
    const layout = await getLayout(params.layoutId);
    return { layout };
  },
});

// ─── Edge Overlay ─────────────────────────────────────────────────────────────

type EdgeHandle = "left" | "right" | "top" | "bottom";

interface EdgeOverlayProps {
  edges: [number, number, number, number]; // [left, right, bottom, top] ft
  dimensions: [number, number]; // [width_ft, height_ft]
  onChange: (edges: [number, number, number, number]) => void;
}

function EdgeOverlay({ edges, dimensions, onChange }: EdgeOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    handle: EdgeHandle;
    startNorm: number;
    startEdges: [number, number, number, number];
  } | null>(null);

  const [widthFt, heightFt] = dimensions;
  const [leftFt, rightFt, bottomFt, topFt] = edges;

  // Convert feet → normalized (0–1), y-axis flipped (ft y=0 is bottom)
  const normLeft = leftFt / widthFt;
  const normRight = rightFt / widthFt;
  const normTop = 1 - topFt / heightFt; // visual top
  const normBottom = 1 - bottomFt / heightFt; // visual bottom

  const boxStyle: React.CSSProperties = {
    position: "absolute",
    left: `${normLeft * 100}%`,
    top: `${normTop * 100}%`,
    width: `${(normRight - normLeft) * 100}%`,
    height: `${(normBottom - normTop) * 100}%`,
    border: "2px solid var(--cyan)",
    boxSizing: "border-box",
    pointerEvents: "none",
  };

  const handleMouseDown = useCallback(
    (handle: EdgeHandle, e: React.MouseEvent) => {
      e.preventDefault();
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const isHorizontal = handle === "left" || handle === "right";
      const startNorm = isHorizontal
        ? (e.clientX - rect.left) / rect.width
        : (e.clientY - rect.top) / rect.height;
      dragRef.current = {
        handle,
        startNorm,
        startEdges: [...edges] as [number, number, number, number],
      };
    },
    [edges],
  );

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      const drag = dragRef.current;
      const container = containerRef.current;
      if (!drag || !container) return;
      const rect = container.getBoundingClientRect();
      const { handle, startNorm, startEdges } = drag;
      const isHorizontal = handle === "left" || handle === "right";
      const normNow = isHorizontal
        ? (e.clientX - rect.left) / rect.width
        : (e.clientY - rect.top) / rect.height;
      const delta = normNow - startNorm;

      const next: [number, number, number, number] = [...startEdges] as [
        number,
        number,
        number,
        number,
      ];
      const minGap = 0.5; // minimum 0.5 ft gap between opposite edges
      if (handle === "left") {
        next[0] = Math.max(
          0,
          Math.min(startEdges[1] - minGap, startEdges[0] + delta * widthFt),
        );
      } else if (handle === "right") {
        next[1] = Math.max(
          startEdges[0] + minGap,
          Math.min(widthFt, startEdges[1] + delta * widthFt),
        );
      } else if (handle === "top") {
        // visual top drag → changes topFt (flipped)
        next[3] = Math.max(
          startEdges[2] + minGap,
          Math.min(heightFt, startEdges[3] - delta * heightFt),
        );
      } else if (handle === "bottom") {
        // visual bottom drag → changes bottomFt (flipped)
        next[2] = Math.max(
          0,
          Math.min(startEdges[3] - minGap, startEdges[2] - delta * heightFt),
        );
      }
      onChange(next);
    };

    const onMouseUp = () => {
      dragRef.current = null;
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [onChange, widthFt, heightFt]);

  const handleStyle = (position: React.CSSProperties): React.CSSProperties => ({
    position: "absolute",
    width: "12px",
    height: "12px",
    background: "var(--cyan)",
    border: "2px solid #09090b",
    borderRadius: "2px",
    zIndex: 10,
    ...position,
  });

  return (
    <div ref={containerRef} style={{ position: "absolute", inset: 0 }}>
      {/* Shaded area outside selection */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(0,0,0,0.45)",
          pointerEvents: "none",
        }}
      />
      {/* Clear window */}
      <div
        style={{
          position: "absolute",
          left: `${normLeft * 100}%`,
          top: `${normTop * 100}%`,
          width: `${(normRight - normLeft) * 100}%`,
          height: `${(normBottom - normTop) * 100}%`,
          boxShadow: "0 0 0 9999px rgba(0,0,0,0.45)",
          background: "transparent",
          pointerEvents: "none",
        }}
      />
      {/* Border */}
      <div style={boxStyle} />

      {/* Left handle */}
      <div
        onMouseDown={(e) => handleMouseDown("left", e)}
        style={{
          ...handleStyle({
            top: `${((normTop + normBottom) / 2) * 100}%`,
            left: `${normLeft * 100}%`,
            transform: "translate(-50%, -50%)",
          }),
          cursor: "ew-resize",
        }}
      />
      {/* Right handle */}
      <div
        onMouseDown={(e) => handleMouseDown("right", e)}
        style={{
          ...handleStyle({
            top: `${((normTop + normBottom) / 2) * 100}%`,
            left: `${normRight * 100}%`,
            transform: "translate(-50%, -50%)",
          }),
          cursor: "ew-resize",
        }}
      />
      {/* Top handle */}
      <div
        onMouseDown={(e) => handleMouseDown("top", e)}
        style={{
          ...handleStyle({
            top: `${normTop * 100}%`,
            left: `${((normLeft + normRight) / 2) * 100}%`,
            transform: "translate(-50%, -50%)",
          }),
          cursor: "ns-resize",
        }}
      />
      {/* Bottom handle */}
      <div
        onMouseDown={(e) => handleMouseDown("bottom", e)}
        style={{
          ...handleStyle({
            top: `${normBottom * 100}%`,
            left: `${((normLeft + normRight) / 2) * 100}%`,
            transform: "translate(-50%, -50%)",
          }),
          cursor: "ns-resize",
        }}
      />
    </div>
  );
}

// ─── SizesPage ────────────────────────────────────────────────────────────────

function SizesPage() {
  const navigate = useNavigate();
  const { isSignedIn, isLoaded } = useUser();
  const { layout: initialLayout } = Route.useLoaderData() as {
    layout: LayoutDetail;
  };
  const layoutId = initialLayout.metadata.id;
  const dims = initialLayout.metadata.dimensions as [number, number];
  const [widthFt, heightFt] = dims;

  const [sizes, setSizes] = useState<SizeMetadata[]>(
    initialLayout.metadata.sizes,
  );
  const [edges, setEdges] = useState<[number, number, number, number]>([
    0,
    widthFt,
    0,
    heightFt,
  ]);
  const [name, setName] = useState("");
  const [kickboard, setKickboard] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const photoUrl = getLayoutPhotoUrl(layoutId);

  const handleEdgeInput = useCallback((index: 0 | 1 | 2 | 3, value: string) => {
    const num = parseFloat(value);
    if (isNaN(num) && value != "") return;
    setEdges((prev) => {
      const next: [number, number, number, number] = [...prev] as [
        number,
        number,
        number,
        number,
      ];
      next[index] = num;
      return next;
    });
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!name.trim()) {
        setError("Size name is required");
        return;
      }
      const [l, r, b, t] = edges;
      if (r <= l || t <= b) {
        setError(
          "Invalid edge values — right must exceed left, top must exceed bottom",
        );
        return;
      }
      setIsSubmitting(true);
      setError(null);
      try {
        await createSize(layoutId, { name: name.trim(), edges, kickboard });
        // Refresh sizes list
        const updated = await getLayout(layoutId);
        setSizes(updated.metadata.sizes);
        setName("");
        setKickboard(false);
        setEdges([0, widthFt, 0, heightFt]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to create size");
      } finally {
        setIsSubmitting(false);
      }
    },
    [layoutId, name, edges, kickboard, widthFt, heightFt],
  );

  const handleDelete = useCallback(
    async (sizeId: string) => {
      try {
        await deleteSize(layoutId, sizeId);
        setSizes((prev) => prev.filter((s) => s.id !== sizeId));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to delete size");
      }
    },
    [layoutId],
  );

  if (!isLoaded || !isSignedIn) {
    return (
      <>
        <style>{GLOBAL_STYLES}</style>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "var(--bg)",
            color: "var(--text-muted)",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.75rem",
          }}
        >
          Sign in to manage sizes.
        </div>
      </>
    );
  }

  const inputStyle: React.CSSProperties = {
    width: "100%",
    background: "var(--surface2)",
    color: "var(--text-primary)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "8px 12px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.7rem",
    outline: "none",
    boxSizing: "border-box",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.55rem",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    color: "var(--text-muted)",
    marginBottom: "6px",
  };

  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          background: "var(--bg)",
          color: "var(--text-primary)",
        }}
      >
        {/* Header */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            padding: "0 20px",
            height: "48px",
            flexShrink: 0,
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
            zIndex: 10,
          }}
        >
          <button
            onClick={() =>
              navigate({ to: "/$layoutId/set", params: { layoutId } })
            }
            style={{
              display: "flex",
              alignItems: "center",
              gap: "5px",
              background: "transparent",
              border: "none",
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.65rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              cursor: "pointer",
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--cyan)")}
            onMouseLeave={(e) =>
              (e.currentTarget.style.color = "var(--text-muted)")
            }
          >
            <ArrowLeft size={12} /> Back
          </button>
          <div
            style={{
              width: "1px",
              height: "16px",
              background: "var(--border)",
            }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <div
              style={{
                width: "2px",
                height: "14px",
                background: "var(--cyan)",
              }}
            />
            <span className="bz-oswald" style={{ fontSize: "0.8rem" }}>
              {initialLayout.metadata.name}
            </span>
            <span
              className="bz-mono"
              style={{
                fontSize: "0.55rem",
                color: "var(--text-dim)",
                letterSpacing: "0.1em",
              }}
            >
              / SIZES
            </span>
          </div>
        </header>

        {/* Body */}
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          {/* Left: image + overlay */}
          <div
            style={{
              flex: 1,
              position: "relative",
              background: "#000",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              minHeight: "400px",
            }}
          >
            <div
              style={{
                position: "relative",
                display: "inline-block",
                maxWidth: "100%",
                maxHeight: "100%",
              }}
            >
              <img
                src={photoUrl}
                alt={initialLayout.metadata.name}
                style={{
                  display: "block",
                  maxWidth: "100%",
                  maxHeight: "calc(100vh - 48px)",
                  objectFit: "contain",
                  userSelect: "none",
                }}
                draggable={false}
              />
              <EdgeOverlay
                edges={edges}
                dimensions={dims}
                onChange={setEdges}
              />
            </div>
          </div>

          {/* Right: form + size list */}
          <div
            style={{
              width: "300px",
              flexShrink: 0,
              display: "flex",
              flexDirection: "column",
              background: "var(--surface)",
              borderLeft: "1px solid var(--border)",
              overflow: "auto",
            }}
          >
            <form
              onSubmit={handleSubmit}
              style={{
                padding: "20px",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <p
                className="bz-oswald"
                style={{
                  fontSize: "0.8rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  marginBottom: "16px",
                }}
              >
                Add Size
              </p>

              {error && (
                <div
                  className="bz-mono"
                  style={{
                    marginBottom: "14px",
                    padding: "8px 10px",
                    background: "rgba(248,113,113,0.08)",
                    border: "1px solid rgba(248,113,113,0.2)",
                    borderRadius: "var(--radius)",
                    fontSize: "0.6rem",
                    color: "#f87171",
                  }}
                >
                  {error}
                </div>
              )}

              {/* Name */}
              <div style={{ marginBottom: "14px" }}>
                <label style={labelStyle}>
                  Name <span style={{ color: "#f87171" }}>*</span>
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Full Panel"
                  style={inputStyle}
                  onFocus={(e) =>
                    (e.currentTarget.style.borderColor = "var(--border-active)")
                  }
                  onBlur={(e) =>
                    (e.currentTarget.style.borderColor = "var(--border)")
                  }
                />
              </div>

              {/* Edge inputs */}
              <div style={{ marginBottom: "14px" }}>
                <label style={labelStyle}>
                  Edges (feet) — left · right · bottom · top
                </label>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "6px",
                  }}
                >
                  {(["Left", "Right", "Bottom", "Top"] as const).map(
                    (label, i) => (
                      <div key={label}>
                        <label style={{ ...labelStyle, marginBottom: "3px" }}>
                          {label}
                        </label>
                        <input
                          type="number"
                          value={edges[i]}
                          step="0.01"
                          min={i === 0 || i === 2 ? 0 : undefined}
                          max={
                            i === 1 ? widthFt : i === 3 ? heightFt : undefined
                          }
                          onChange={(e) =>
                            handleEdgeInput(i as 0 | 1 | 2 | 3, e.target.value)
                          }
                          style={{ ...inputStyle, padding: "6px 8px" }}
                          onFocus={(e) =>
                            (e.currentTarget.style.borderColor =
                              "var(--border-active)")
                          }
                          onBlur={(e) =>
                            (e.currentTarget.style.borderColor =
                              "var(--border)")
                          }
                        />
                      </div>
                    ),
                  )}
                </div>
              </div>

              {/* Kickboard */}
              <div style={{ marginBottom: "20px" }}>
                <label style={labelStyle}>Kickboard</label>
                <button
                  type="button"
                  onClick={() => setKickboard((v) => !v)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "8px 12px",
                    background: kickboard
                      ? "var(--cyan-dim)"
                      : "var(--surface2)",
                    border: `1px solid ${kickboard ? "var(--border-active)" : "var(--border)"}`,
                    borderRadius: "var(--radius)",
                    cursor: "pointer",
                    transition: "all 0.15s",
                    width: "100%",
                  }}
                >
                  <div
                    style={{
                      width: "28px",
                      height: "16px",
                      borderRadius: "8px",
                      background: kickboard
                        ? "var(--cyan)"
                        : "rgba(255,255,255,0.15)",
                      position: "relative",
                      flexShrink: 0,
                      transition: "background 0.15s",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        top: "2px",
                        left: kickboard ? "14px" : "2px",
                        width: "12px",
                        height: "12px",
                        borderRadius: "50%",
                        background: "#fff",
                        transition: "left 0.15s",
                      }}
                    />
                  </div>
                  <span
                    className="bz-mono"
                    style={{
                      fontSize: "0.6rem",
                      color: kickboard ? "var(--cyan)" : "var(--text-muted)",
                      transition: "color 0.15s",
                    }}
                  >
                    {kickboard ? "Yes — has kickboard" : "No kickboard"}
                  </span>
                </button>
              </div>

              <button
                type="submit"
                disabled={isSubmitting || !name.trim()}
                className="bz-oswald"
                style={{
                  width: "100%",
                  padding: "9px 16px",
                  background:
                    isSubmitting || !name.trim()
                      ? "var(--surface2)"
                      : "var(--cyan)",
                  color:
                    isSubmitting || !name.trim()
                      ? "var(--text-muted)"
                      : "#09090b",
                  border: "none",
                  borderRadius: "var(--radius)",
                  fontSize: "0.8rem",
                  letterSpacing: "0.1em",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  cursor:
                    isSubmitting || !name.trim() ? "not-allowed" : "pointer",
                  transition: "all 0.15s",
                }}
              >
                {isSubmitting ? "Adding…" : "Add Size"}
              </button>
            </form>

            {/* Existing sizes */}
            <div style={{ padding: "16px 20px", flex: 1 }}>
              <p
                className="bz-oswald"
                style={{
                  fontSize: "0.75rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  marginBottom: "12px",
                  color: "var(--text-muted)",
                }}
              >
                Existing Sizes ({sizes.length})
              </p>
              {sizes.length === 0 ? (
                <p
                  className="bz-mono"
                  style={{ fontSize: "0.6rem", color: "var(--text-dim)" }}
                >
                  No sizes yet.
                </p>
              ) : (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  {sizes.map((size) => (
                    <div
                      key={size.id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "8px 12px",
                        background: "var(--surface2)",
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius)",
                      }}
                    >
                      <div>
                        <div
                          className="bz-mono"
                          style={{
                            fontSize: "0.65rem",
                            color: "var(--text-primary)",
                            marginBottom: "2px",
                          }}
                        >
                          {size.name}
                        </div>
                        <div
                          className="bz-mono"
                          style={{
                            fontSize: "0.55rem",
                            color: "var(--text-dim)",
                          }}
                        >
                          [{size.edges.map((v) => v.toFixed(1)).join(", ")}]
                          {size.kickboard && " · kickboard"}
                        </div>
                      </div>
                      <button
                        onClick={() => handleDelete(size.id)}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          width: "24px",
                          height: "24px",
                          background: "transparent",
                          border: "none",
                          color: "var(--text-dim)",
                          cursor: "pointer",
                          borderRadius: "var(--radius)",
                          flexShrink: 0,
                          transition: "color 0.15s",
                        }}
                        onMouseEnter={(e) =>
                          (e.currentTarget.style.color = "#ef4444")
                        }
                        onMouseLeave={(e) =>
                          (e.currentTarget.style.color = "var(--text-dim)")
                        }
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
