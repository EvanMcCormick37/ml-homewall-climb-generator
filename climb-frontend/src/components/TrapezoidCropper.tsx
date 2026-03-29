import { useRef, useCallback } from "react";

/**
 * 8-float array: [tlx, tly, trx, try, blx, bly, brx, bry]
 * All values normalized to [0, 1] in image space.
 *
 * Corner semantics (matching wall-coordinate convention):
 *   TL (top-left  in image) → (0,   H_ft) in wall coords
 *   TR (top-right in image) → (W_ft, H_ft)
 *   BL (bot-left  in image) → (0,   0)
 *   BR (bot-right in image) → (W_ft, 0)
 */
export type TrapCorners = [
  number, number, // TL x, y
  number, number, // TR x, y
  number, number, // BL x, y
  number, number, // BR x, y
];

export interface TrapezoidCropperProps {
  imageUrl: string;
  corners: TrapCorners;
  onChange: (corners: TrapCorners) => void;
}

type CornerIndex = 0 | 1 | 2 | 3; // TL, TR, BL, BR

const CORNER_LABELS: Record<CornerIndex, string> = {
  0: "TL",
  1: "TR",
  2: "BL",
  3: "BR",
};

const HANDLE_RADIUS = 10; // px, visual only

export default function TrapezoidCropper({
  imageUrl,
  corners,
  onChange,
}: TrapezoidCropperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ cornerIdx: CornerIndex } | null>(null);

  // Unpack corner pairs for convenience
  const pts: [number, number][] = [
    [corners[0], corners[1]], // TL
    [corners[2], corners[3]], // TR
    [corners[4], corners[5]], // BL
    [corners[6], corners[7]], // BR
  ];

  // SVG polygon points string (as percentages inside a viewBox="0 0 100 100")
  const polygonPoints = pts
    .map(([x, y]) => `${x * 100},${y * 100}`)
    .join(" ");

  const getNormCoords = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return [0, 0];
      return [
        Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
        Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
      ];
    },
    []
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent, idx: CornerIndex) => {
      e.preventDefault();
      e.stopPropagation();
      dragRef.current = { cornerIdx: idx };
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    []
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current) return;
      const [nx, ny] = getNormCoords(e.clientX, e.clientY);
      const idx = dragRef.current.cornerIdx;
      const next = [...corners] as TrapCorners;
      next[idx * 2] = nx;
      next[idx * 2 + 1] = ny;
      onChange(next);
    },
    [corners, onChange, getNormCoords]
  );

  const handlePointerUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full select-none overflow-hidden rounded-lg bg-zinc-900"
      style={{ touchAction: "none" }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      {/* Base image */}
      <img
        src={imageUrl}
        alt="Wall to map"
        className="block w-full h-auto"
        draggable={false}
      />

      {/* SVG overlay — sits on top of image, covers it exactly */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ pointerEvents: "none" }}
      >
        {/* Quad outline */}
        <polygon
          points={polygonPoints}
          fill="none"
          stroke="#06b6d4"
          strokeWidth="0.5"
          strokeDasharray="2 1.5"
        />

        {/* Grid lines inside the quad (rule-of-thirds approximation) */}
        {([1 / 3, 2 / 3] as const).map((t) => {
          // Interpolate between left edge (TL→BL) and right edge (TR→BR)
          const lx = pts[0][0] + t * (pts[2][0] - pts[0][0]);
          const ly = pts[0][1] + t * (pts[2][1] - pts[0][1]);
          const rx = pts[1][0] + t * (pts[3][0] - pts[1][0]);
          const ry = pts[1][1] + t * (pts[3][1] - pts[1][1]);
          // Interpolate between top edge (TL→TR) and bottom edge (BL→BR)
          const tx = pts[0][0] + t * (pts[1][0] - pts[0][0]);
          const ty = pts[0][1] + t * (pts[1][1] - pts[0][1]);
          const bx = pts[2][0] + t * (pts[3][0] - pts[2][0]);
          const by = pts[2][1] + t * (pts[3][1] - pts[2][1]);
          return (
            <g key={t}>
              <line
                x1={lx * 100} y1={ly * 100}
                x2={rx * 100} y2={ry * 100}
                stroke="rgba(6,182,212,0.3)" strokeWidth="0.3"
              />
              <line
                x1={tx * 100} y1={ty * 100}
                x2={bx * 100} y2={by * 100}
                stroke="rgba(6,182,212,0.3)" strokeWidth="0.3"
              />
            </g>
          );
        })}
      </svg>

      {/* Draggable corner handles — rendered as absolutely-positioned divs */}
      {(Object.keys(CORNER_LABELS) as unknown as CornerIndex[]).map((rawIdx) => {
        const idx = Number(rawIdx) as CornerIndex;
        const [nx, ny] = pts[idx];
        return (
          <div
            key={idx}
            onPointerDown={(e) => handlePointerDown(e, idx)}
            style={{
              position: "absolute",
              left: `${nx * 100}%`,
              top: `${ny * 100}%`,
              transform: "translate(-50%, -50%)",
              width: HANDLE_RADIUS * 2,
              height: HANDLE_RADIUS * 2,
              borderRadius: "50%",
              background: "#09090b",
              border: "2px solid #06b6d4",
              cursor: "grab",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 10,
              userSelect: "none",
              touchAction: "none",
              boxShadow: "0 0 0 3px rgba(6,182,212,0.25)",
            }}
          >
            <span
              style={{
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.45rem",
                color: "#06b6d4",
                letterSpacing: "0.05em",
                pointerEvents: "none",
              }}
            >
              {CORNER_LABELS[idx]}
            </span>
          </div>
        );
      })}

      {/* Instructions */}
      <div
        className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded"
        style={{
          background: "rgba(0,0,0,0.75)",
          color: "var(--text-muted, #71717a)",
          fontFamily: "'Space Mono', monospace",
          fontSize: "0.65rem",
          letterSpacing: "0.06em",
          whiteSpace: "nowrap",
          pointerEvents: "none",
        }}
      >
        Drag TL · TR · BL · BR handles to the wall corners
      </div>
    </div>
  );
}
