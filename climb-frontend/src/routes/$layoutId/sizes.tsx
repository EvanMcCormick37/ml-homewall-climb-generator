import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useRef, useCallback, useEffect } from "react";
import { useUser } from "@clerk/clerk-react";
import { getLayout, fetchLayoutPhoto, updateLayout } from "@/api/layouts";
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
  /** Where the image edges sit in wall-coordinate ft. Defaults to [0,W,0,H]. */
  imageEdges?: [number, number, number, number];
}

function EdgeOverlay({ edges, dimensions, onChange, imageEdges }: EdgeOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    handle: EdgeHandle;
    startNorm: number;
    startEdges: [number, number, number, number];
  } | null>(null);

  const [widthFt, heightFt] = dimensions;
  const [leftFt, rightFt, bottomFt, topFt] = edges;

  // Image coordinate space — falls back to full wall if no imageEdges set
  const [imgL, imgR, imgB, imgT] = imageEdges ?? [0, widthFt, 0, heightFt];
  const hspan = imgR - imgL;
  const vspan = imgT - imgB;

  // Convert size feet → image-normalized (0–1), y-axis flipped
  const normLeft   = (leftFt   - imgL) / hspan;
  const normRight  = (rightFt  - imgL) / hspan;
  const normTop    = (imgT - topFt)    / vspan; // visual top
  const normBottom = (imgT - bottomFt) / vspan; // visual bottom

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
        next[0] = Math.min(startEdges[1] - minGap, startEdges[0] + delta * hspan);
      } else if (handle === "right") {
        next[1] = Math.max(startEdges[0] + minGap, startEdges[1] + delta * hspan);
      } else if (handle === "top") {
        // visual top drag → changes topFt (flipped)
        next[3] = Math.max(startEdges[2] + minGap, startEdges[3] - delta * vspan);
      } else if (handle === "bottom") {
        // visual bottom drag → changes bottomFt (flipped)
        next[2] = Math.min(startEdges[3] - minGap, startEdges[2] - delta * vspan);
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
  }, [onChange, widthFt, heightFt, hspan, vspan]);

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

// ─── ImageAlignmentOverlay ────────────────────────────────────────────────────
//
// Shows a draggable box on the image representing where the wall's [0,W,0,H]
// boundary falls. Dragging the box edges updates imageEdges [L,R,B,T] in ft.
//
// Math: given box position as image-fractions (0-1):
//   normL = fraction from image left where wall-left (0) falls
//   normR = fraction from image left where wall-right (W) falls
//   → imgL = -normL * W / (normR - normL)
//   → imgR =  (1-normL) * W / (normR - normL)
// Similarly for vertical (image top = y-fraction 0, y increases downward):
//   normT = image-fraction where wall-top (H) falls
//   normB = image-fraction where wall-bottom (0) falls
//   → imgT = normB * H / (normB - normT)
//   → imgB = (normB-1) * H / (normB - normT)

interface ImageAlignmentOverlayProps {
  imageEdges: [number, number, number, number]; // [imgL, imgR, imgB, imgT] in ft
  dimensions: [number, number]; // [width_ft, height_ft] wall dims
  onChange: (edges: [number, number, number, number]) => void;
}

function imageEdgesToNorm(
  imageEdges: [number, number, number, number],
  widthFt: number,
  heightFt: number,
): { normL: number; normR: number; normT: number; normB: number } {
  const [imgL, imgR, imgB, imgT] = imageEdges;
  const hspan = imgR - imgL;
  const vspan = imgT - imgB;
  return {
    normL: (0 - imgL) / hspan, // where wall-left (0 ft) falls in image
    normR: (widthFt - imgL) / hspan, // where wall-right (W ft) falls in image
    normT: (imgT - heightFt) / vspan, // where wall-top (H ft) falls in image (y-down)
    normB: (imgT - 0) / vspan, // where wall-bottom (0 ft) falls in image (y-down)
  };
}

function normToImageEdges(
  normL: number,
  normR: number,
  normT: number,
  normB: number,
  widthFt: number,
  heightFt: number,
): [number, number, number, number] {
  const hspan = widthFt / (normR - normL);
  const imgL = -normL * hspan;
  const imgR = imgL + hspan;
  const vspan = heightFt / (normB - normT);
  const imgT = normB * vspan;
  const imgB = imgT - vspan;
  return [imgL, imgR, imgB, imgT];
}

function ImageAlignmentOverlay({
  imageEdges,
  dimensions,
  onChange,
}: ImageAlignmentOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    handle: EdgeHandle;
    startNorm: number;
    startNorms: { normL: number; normR: number; normT: number; normB: number };
  } | null>(null);
  const panRef = useRef<{
    startX: number;
    startY: number;
    startNorms: { normL: number; normR: number; normT: number; normB: number };
  } | null>(null);

  const [widthFt, heightFt] = dimensions;
  const { normL, normR, normT, normB } = imageEdgesToNorm(
    imageEdges,
    widthFt,
    heightFt,
  );

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
        startNorms: { normL, normR, normT, normB },
      };
    },
    [normL, normR, normT, normB],
  );

  const handlePanMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      panRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startNorms: { normL, normR, normT, normB },
      };
    },
    [normL, normR, normT, normB],
  );

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();

      // Pan (whole-box drag)
      const pan = panRef.current;
      if (pan) {
        const dx = (e.clientX - pan.startX) / rect.width;
        const dy = (e.clientY - pan.startY) / rect.height;
        const { normL: nL, normR: nR, normT: nT, normB: nB } = pan.startNorms;
        onChange(normToImageEdges(nL + dx, nR + dx, nT + dy, nB + dy, widthFt, heightFt));
        return;
      }

      // Edge handle drag
      const drag = dragRef.current;
      if (!drag) return;
      const { handle, startNorm, startNorms } = drag;
      const isHorizontal = handle === "left" || handle === "right";
      const normNow = isHorizontal
        ? (e.clientX - rect.left) / rect.width
        : (e.clientY - rect.top) / rect.height;
      const delta = normNow - startNorm;
      const minGap = 0.05;

      let { normL: nL, normR: nR, normT: nT, normB: nB } = startNorms;
      if (handle === "left") {
        nL = Math.min(startNorms.normR - minGap, startNorms.normL + delta);
      } else if (handle === "right") {
        nR = Math.max(startNorms.normL + minGap, startNorms.normR + delta);
      } else if (handle === "top") {
        nT = Math.min(startNorms.normB - minGap, startNorms.normT + delta);
      } else if (handle === "bottom") {
        nB = Math.max(startNorms.normT + minGap, startNorms.normB + delta);
      }
      onChange(normToImageEdges(nL, nR, nT, nB, widthFt, heightFt));
    };
    const onMouseUp = () => {
      dragRef.current = null;
      panRef.current = null;
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

  const boxStyle: React.CSSProperties = {
    position: "absolute",
    left: `${normL * 100}%`,
    top: `${normT * 100}%`,
    width: `${(normR - normL) * 100}%`,
    height: `${(normB - normT) * 100}%`,
    border: "2px solid var(--cyan)",
    boxSizing: "border-box",
    pointerEvents: "none",
  };

  return (
    <div ref={containerRef} style={{ position: "absolute", inset: 0 }}>
      {/* Shaded area outside wall boundary */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(0,0,0,0.45)",
          pointerEvents: "none",
        }}
      />
      {/* Clear window — the wall boundary region (draggable to pan) */}
      <div
        onMouseDown={handlePanMouseDown}
        style={{
          position: "absolute",
          left: `${normL * 100}%`,
          top: `${normT * 100}%`,
          width: `${(normR - normL) * 100}%`,
          height: `${(normB - normT) * 100}%`,
          boxShadow: "0 0 0 9999px rgba(0,0,0,0.45)",
          background: "transparent",
          cursor: "move",
        }}
      />
      {/* Border */}
      <div style={boxStyle} />
      {/* Left handle */}
      <div
        onMouseDown={(e) => handleMouseDown("left", e)}
        style={{
          ...handleStyle({
            top: `${((normT + normB) / 2) * 100}%`,
            left: `${normL * 100}%`,
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
            top: `${((normT + normB) / 2) * 100}%`,
            left: `${normR * 100}%`,
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
            top: `${normT * 100}%`,
            left: `${((normL + normR) / 2) * 100}%`,
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
            top: `${normB * 100}%`,
            left: `${((normL + normR) / 2) * 100}%`,
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

  // Image alignment state
  const defaultImageEdges: [number, number, number, number] = [
    0,
    widthFt,
    0,
    heightFt,
  ];
  const [imageEdges, setImageEdges] = useState<
    [number, number, number, number]
  >(
    (initialLayout.metadata.image_edges as
      | [number, number, number, number]
      | null) ?? defaultImageEdges,
  );
  const [isSavingAlignment, setIsSavingAlignment] = useState(false);
  const [alignmentError, setAlignmentError] = useState<string | null>(null);
  const [alignmentSaved, setAlignmentSaved] = useState(false);

  // Active right-panel tab: "sizes" | "alignment"
  const [activeTab, setActiveTab] = useState<"sizes" | "alignment">("sizes");

  const handleImageEdgeInput = useCallback(
    (index: 0 | 1 | 2 | 3, value: string) => {
      const num = parseFloat(value);
      if (isNaN(num)) return "";
      setImageEdges((prev) => {
        const next = [...prev] as [number, number, number, number];
        next[index] = isNaN(num) ? 0 : num;
        return next;
      });
    },
    [],
  );

  const handleSaveAlignment = useCallback(async () => {
    setIsSavingAlignment(true);
    setAlignmentError(null);
    setAlignmentSaved(false);
    try {
      await updateLayout(layoutId, { image_edges: imageEdges });
      setAlignmentSaved(true);
      setTimeout(() => setAlignmentSaved(false), 2000);
    } catch (err) {
      setAlignmentError(
        err instanceof Error ? err.message : "Failed to save alignment",
      );
    } finally {
      setIsSavingAlignment(false);
    }
  }, [layoutId, imageEdges]);

  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  useEffect(() => {
    let url: string | null = null;
    fetchLayoutPhoto(layoutId).then((u) => { url = u; setPhotoUrl(u); });
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [layoutId]);

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
              {/* Hold circles overlay */}
              {initialLayout.holds.length > 0 &&
                (() => {
                  const [imgL, imgR, imgB, imgT] = imageEdges;
                  const hspan = imgR - imgL;
                  const vspan = imgT - imgB;
                  return (
                    <svg
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        pointerEvents: "none",
                      }}
                    >
                      {initialLayout.holds.map((hold) => {
                        const cx = ((hold.x - imgL) / hspan) * 100;
                        const cy = ((imgT - hold.y) / vspan) * 100;
                        return (
                          <circle
                            key={hold.hold_index}
                            cx={`${cx}%`}
                            cy={`${cy}%`}
                            r={8}
                            fill="none"
                            stroke="#00b679"
                            strokeWidth={2}
                            opacity={0.5}
                          />
                        );
                      })}
                    </svg>
                  );
                })()}
              {activeTab === "sizes" ? (
                <EdgeOverlay
                  edges={edges}
                  dimensions={dims}
                  onChange={setEdges}
                  imageEdges={imageEdges}
                />
              ) : (
                <ImageAlignmentOverlay
                  imageEdges={imageEdges}
                  dimensions={dims}
                  onChange={setImageEdges}
                />
              )}
            </div>
          </div>

          {/* Right: tabs + form + size list */}
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
            {/* Tab bar */}
            <div
              style={{
                display: "flex",
                borderBottom: "1px solid var(--border)",
                flexShrink: 0,
              }}
            >
              {(["sizes", "alignment"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className="bz-mono"
                  style={{
                    flex: 1,
                    padding: "10px",
                    background: "transparent",
                    border: "none",
                    borderBottom: `2px solid ${activeTab === tab ? "var(--cyan)" : "transparent"}`,
                    color:
                      activeTab === tab ? "var(--cyan)" : "var(--text-muted)",
                    fontSize: "0.55rem",
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    cursor: "pointer",
                    transition: "color 0.15s",
                  }}
                >
                  {tab === "sizes" ? "Sizes" : "Image Align"}
                </button>
              ))}
            </div>

            {activeTab === "alignment" ? (
              /* ── Image Alignment Panel ─────────────────────────────────── */
              <div style={{ padding: "20px", flex: 1 }}>
                <p
                  className="bz-oswald"
                  style={{
                    fontSize: "0.8rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.06em",
                    marginBottom: "8px",
                  }}
                >
                  Image Alignment
                </p>
                <p
                  className="bz-mono"
                  style={{
                    fontSize: "0.55rem",
                    color: "var(--text-dim)",
                    lineHeight: 1.6,
                    marginBottom: "16px",
                  }}
                >
                  Drag the box on the image to mark where the wall's physical
                  boundary falls in the photo. Holds and size overlays will
                  shift to match.
                </p>

                {alignmentError && (
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
                    {alignmentError}
                  </div>
                )}

                {/* Numeric inputs for image_edges [L, R, B, T] */}
                <div style={{ marginBottom: "14px" }}>
                  <label style={labelStyle}>
                    Image edges (ft) — left · right · bottom · top
                  </label>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "6px",
                    }}
                  >
                    {(["Left", "Right", "Bottom", "Top"] as const).map(
                      (lbl, i) => (
                        <div key={lbl}>
                          <label style={{ ...labelStyle, marginBottom: "3px" }}>
                            {lbl}
                          </label>
                          <input
                            type="number"
                            value={imageEdges[i]}
                            step="0.1"
                            onChange={(e) =>
                              handleImageEdgeInput(
                                i as 0 | 1 | 2 | 3,
                                e.target.value,
                              )
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

                {/* Reset button */}
                <button
                  type="button"
                  onClick={() => setImageEdges(defaultImageEdges)}
                  className="bz-mono"
                  style={{
                    width: "100%",
                    padding: "7px 12px",
                    marginBottom: "10px",
                    background: "var(--surface2)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    color: "var(--text-muted)",
                    fontSize: "0.6rem",
                    letterSpacing: "0.08em",
                    cursor: "pointer",
                    transition: "color 0.15s",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.color = "var(--text-primary)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.color = "var(--text-muted)")
                  }
                >
                  Reset to full image
                </button>

                {/* Save button */}
                <button
                  type="button"
                  onClick={handleSaveAlignment}
                  disabled={isSavingAlignment}
                  className="bz-oswald"
                  style={{
                    width: "100%",
                    padding: "9px 16px",
                    background: alignmentSaved
                      ? "rgba(6,182,212,0.2)"
                      : isSavingAlignment
                        ? "var(--surface2)"
                        : "var(--cyan)",
                    color: alignmentSaved
                      ? "var(--cyan)"
                      : isSavingAlignment
                        ? "var(--text-muted)"
                        : "#09090b",
                    border: alignmentSaved
                      ? "1px solid var(--border-active)"
                      : "none",
                    borderRadius: "var(--radius)",
                    fontSize: "0.8rem",
                    letterSpacing: "0.1em",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    cursor: isSavingAlignment ? "not-allowed" : "pointer",
                    transition: "all 0.15s",
                  }}
                >
                  {alignmentSaved
                    ? "Saved"
                    : isSavingAlignment
                      ? "Saving…"
                      : "Save Alignment"}
                </button>
              </div>
            ) : (
              /* ── Sizes Panel ───────────────────────────────────────────── */
              <>
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
                        (e.currentTarget.style.borderColor =
                          "var(--border-active)")
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
                            <label
                              style={{ ...labelStyle, marginBottom: "3px" }}
                            >
                              {label}
                            </label>
                            <input
                              type="number"
                              value={edges[i]}
                              step="0.01"
                              onChange={(e) =>
                                handleEdgeInput(
                                  i as 0 | 1 | 2 | 3,
                                  e.target.value,
                                )
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
                          color: kickboard
                            ? "var(--cyan)"
                            : "var(--text-muted)",
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
                        isSubmitting || !name.trim()
                          ? "not-allowed"
                          : "pointer",
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
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
