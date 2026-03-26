import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect } from "react";
import { getLayout, setLayoutHolds, getLayoutPhotoUrl } from "@/api/layouts";
import { useHolds } from "@/hooks/useHolds";
import { HoldFeaturesSidebar, EnabledFeaturesMenu } from "@/components";
import { Eraser, Hand, Settings, Plus, Edit } from "lucide-react";
import { GLOBAL_STYLES } from "@/styles";
import type { HoldDetail, LayoutDetail, HoldMode, Tag, EnabledFeatures } from "@/types";

export const Route = createFileRoute("/$layoutId/holds")({
  component: HoldsEditorPage,
  loader: async ({ params }) => {
    const layout = await getLayout(params.layoutId);
    return { layout };
  },
});

// ── Types ──────────────────────────────────────────────────────────────────────

type AddDragState = { isDragging: boolean; holdX: number; holdY: number; dragX: number; dragY: number };
type EditDragState = { isDragging: boolean; dragX: number; dragY: number; originalHold?: HoldDetail };
type HoldParams = { pull_x: number; pull_y: number; useability: number; x: number; y: number };

// ── getDragParams ──────────────────────────────────────────────────────────────

function getDragParams(
  add: AddDragState,
  edit: EditDragState,
  toPixelCoords: (h: Pick<HoldDetail, "x" | "y">) => { x: number; y: number },
  calcParams: (hx: number, hy: number, dx: number, dy: number) => HoldParams,
): HoldParams {
  if (add.isDragging)
    return calcParams(add.holdX, add.holdY, add.dragX, add.dragY);
  if (edit.isDragging && edit.originalHold) {
    const p = toPixelCoords(edit.originalHold);
    return calcParams(p.x, p.y, edit.dragX, edit.dragY);
  }
  return { pull_x: 0, pull_y: -1, useability: 0, x: 0, y: 0 };
}

// ── useZoomPan ─────────────────────────────────────────────────────────────────

function useZoomPan(wrapperRef: React.RefObject<HTMLDivElement>) {
  const [viewTransform, setViewTransform] = useState({ zoom: 1, x: 0, y: 0 });

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setViewTransform((prev) => {
        const newZoom = Math.max(0.1, Math.min(10, prev.zoom * factor));
        const scale = newZoom / prev.zoom;
        return { zoom: newZoom, x: mx - (mx - prev.x) * scale, y: my - (my - prev.y) * scale };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [wrapperRef]);

  return { viewTransform, setViewTransform };
}

// ── useHoldHotkeys ─────────────────────────────────────────────────────────────

function useHoldHotkeys(opts: {
  enabledFeatures: EnabledFeatures;
  removeLastHold: () => void;
  setHoldMode: (m: HoldMode) => void;
  setIsAddFoot: React.Dispatch<React.SetStateAction<boolean>>;
  setStickyTag: React.Dispatch<React.SetStateAction<Tag | null>>;
}) {
  const { enabledFeatures, removeLastHold, setHoldMode, setIsAddFoot, setStickyTag } = opts;
  useEffect(() => {
    const TAG_KEYS: Record<string, Tag> = {
      p: "pinch", P: "pinch", m: "macro", M: "macro",
      s: "sloper", S: "sloper", f: "flat", F: "flat", j: "jug", J: "jug",
    };
    const onKeydown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); removeLastHold(); return; }
      switch (e.key) {
        case "1": e.preventDefault(); setHoldMode("add"); break;
        case "2": e.preventDefault(); setHoldMode("edit"); break;
        case "3": e.preventDefault(); setHoldMode("remove"); break;
        case "4": e.preventDefault(); setHoldMode("select"); break;
        case "x": case "X": setIsAddFoot((p) => !p); break;
        default:
          if (enabledFeatures.tags) {
            const tag = TAG_KEYS[e.key];
            if (tag) setStickyTag((p) => (p === tag ? null : tag));
          }
      }
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [enabledFeatures.tags, removeLastHold, setHoldMode, setIsAddFoot, setStickyTag]);
}

// ── useCanvasDraw ──────────────────────────────────────────────────────────────

function useCanvasDraw(
  canvasRef: React.RefObject<HTMLCanvasElement>,
  opts: {
    image: HTMLImageElement | null;
    imageDimensions: { width: number; height: number };
    holds: HoldDetail[];
    mode: HoldMode;
    selectedHold: HoldDetail | null;
    addDragState: AddDragState;
    editDragState: EditDragState;
    enabledFeatures: EnabledFeatures;
    isAddFoot: boolean;
    stickyTag: Tag | null;
    toPixelCoords: (hold: Pick<HoldDetail, "x" | "y">) => { x: number; y: number };
    calculateHoldParams: (hx: number, hy: number, dx: number, dy: number) => HoldParams;
    getHoldColor: (u: number, isFoot: boolean) => string;
  },
) {
  const {
    image, imageDimensions, holds, mode, selectedHold,
    addDragState, editDragState, enabledFeatures, isAddFoot, stickyTag,
    toPixelCoords, calculateHoldParams, getHoldColor,
  } = opts;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas?.getContext("2d");
    if (!image || !ctx) return;

    const { width, height } = imageDimensions;
    canvas.width = width || 800;
    canvas.height = height || 600;

    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const scale = height / 1500;
      const sm = hold.is_foot ? 0.5 * scale : scale;
      const u = hold.useability ?? 0.5;
      const color = getHoldColor(u, hold.is_foot);
      const cs = 4 * sm;
      const as = 2 * sm;

      ctx.globalAlpha = 0.4;

      if (mode === "select" && selectedHold?.hold_index === hold.hold_index) {
        ctx.beginPath();
        ctx.arc(x, y, 20 * sm, 0, Math.PI * 2);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 4 * sm;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(x, y, cs * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = cs;
      ctx.stroke();

      if (enabledFeatures.direction && hold.pull_x != null && hold.pull_y != null) {
        const arrowLen = (10 + 30 * as * u) * sm;
        const ex = x + hold.pull_x * arrowLen;
        const ey = y + hold.pull_y * arrowLen;
        const angle = Math.atan2(hold.pull_y, hold.pull_x);
        const hl = arrowLen / 5;

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = color;
        ctx.lineWidth = as;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(ex + (as / 2) * Math.cos(angle - Math.PI / 4), ey + (as / 2) * Math.sin(angle - Math.PI / 4));
        ctx.lineTo(ex - hl * Math.cos(angle - Math.PI / 4), ey - hl * Math.sin(angle - Math.PI / 4));
        ctx.moveTo(ex + (as / 2) * Math.cos(angle + Math.PI / 4), ey + (as / 2) * Math.sin(angle + Math.PI / 4));
        ctx.lineTo(ex - hl * Math.cos(angle + Math.PI / 4), ey - hl * Math.sin(angle + Math.PI / 4));
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
      if (stickyTag && hold.tags.includes(stickyTag)) {
        ctx.beginPath();
        ctx.arc(x, y, cs * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "#06b6d4";
        ctx.fill();
      }
    });

    // Add-hold drag preview
    if (addDragState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addDragState;
      const params = calculateHoldParams(holdX, holdY, dragX, dragY);
      const sm = isAddFoot ? 0.5 : 1;
      const color = getHoldColor(params.useability, isAddFoot);
      const cs = 6 * sm, as = 4 * sm;

      ctx.beginPath();
      ctx.arc(holdX, holdY, cs * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = cs;
      ctx.stroke();
      ctx.setLineDash([]);

      if (enabledFeatures.direction) {
        ctx.beginPath();
        ctx.moveTo(dragX, dragY);
        ctx.lineTo(holdX, holdY);
        ctx.strokeStyle = color;
        ctx.lineWidth = as;
        ctx.stroke();
      }
    }

    // Edit-hold drag preview
    if (editDragState.isDragging && editDragState.originalHold) {
      const { originalHold, dragX, dragY } = editDragState;
      const pc = toPixelCoords(originalHold);
      const params = calculateHoldParams(pc.x, pc.y, dragX, dragY);
      const sm = originalHold.is_foot ? 0.5 : 1;
      const color = getHoldColor(params.useability, originalHold.is_foot);
      const cs = 6 * sm, as = 4 * sm;

      ctx.beginPath();
      ctx.arc(pc.x, pc.y, cs * 4, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = cs;
      ctx.stroke();
      ctx.setLineDash([]);

      if (enabledFeatures.direction) {
        ctx.beginPath();
        ctx.moveTo(dragX, dragY);
        ctx.lineTo(pc.x, pc.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = as;
        ctx.stroke();
      }
    }
  }, [
    image, imageDimensions, holds, addDragState, editDragState,
    calculateHoldParams, getHoldColor, mode, selectedHold, toPixelCoords,
    enabledFeatures, isAddFoot, stickyTag,
  ]);
}

// ── HoldsHeader ────────────────────────────────────────────────────────────────

function HoldsHeader(props: {
  layoutName: string;
  mode: HoldMode;
  enabledFeatures: EnabledFeatures;
  isAddFoot: boolean;
  showFeatureMenu: boolean;
  isSubmitting: boolean;
  onToggleFeatureMenu: () => void;
  onSetMode: (m: HoldMode) => void;
  onClear: () => void;
  onCancel: () => void;
  onSave: () => void;
}) {
  const { layoutName, mode, enabledFeatures, isAddFoot, showFeatureMenu, isSubmitting,
    onToggleFeatureMenu, onSetMode, onClear, onCancel, onSave } = props;

  const footColor = enabledFeatures.footholds && isAddFoot ? "#9333ea" : "#34d399";
  const modeBtn = (active: boolean, color: string): React.CSSProperties => ({
    padding: "6px 10px", borderRadius: "var(--radius)", border: "none", cursor: "pointer",
    display: "flex", alignItems: "center", gap: "6px", transition: "all 0.15s",
    fontFamily: "'Space Mono', monospace", fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em",
    background: active ? color : "transparent",
    color: active ? "#09090b" : "var(--text-muted)",
  });

  return (
    <header style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 20px", height: "48px", flexShrink: 0,
      background: "var(--surface)", borderBottom: "1px solid var(--border)", zIndex: 10,
    }}>
      <button
        onClick={onToggleFeatureMenu}
        style={{
          padding: "6px 10px", borderRadius: "var(--radius)", border: "none",
          cursor: "pointer", display: "flex", alignItems: "center", gap: "6px", transition: "all 0.15s",
          background: showFeatureMenu ? "var(--cyan-dim)" : "transparent",
          color: showFeatureMenu ? "var(--cyan)" : "var(--text-muted)",
        }}
      >
        <Settings size={14} />
        <span className="bz-mono" style={{ fontSize: "0.55rem", letterSpacing: "0.15em", textTransform: "uppercase" }}>
          Hold Features Settings
        </span>
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{ width: "2px", height: "14px", background: "var(--cyan)" }} />
          <span className="bz-oswald" style={{ fontSize: "0.8rem", color: "var(--text-primary)", letterSpacing: "0.04em" }}>
            {layoutName}
          </span>
          <span className="bz-mono" style={{ fontSize: "0.55rem", color: "var(--text-dim)", letterSpacing: "0.1em" }}>
            / HOLDS
          </span>
        </div>
        <div style={{
          display: "flex", background: "var(--bg)", borderRadius: "var(--radius)",
          padding: "3px", border: "1px solid var(--border)", gap: "2px",
        }}>
          <button onClick={() => onSetMode("add")} style={modeBtn(mode === "add", footColor)}>
            {mode === "add"
              ? <span style={{ display: "flex", alignItems: "center", gap: "4px" }}><Plus size={12} />{enabledFeatures.footholds && isAddFoot ? "FOOT" : "HAND"}</span>
              : <Plus size={13} />}
          </button>
          <button onClick={() => onSetMode("edit")} style={modeBtn(mode === "edit", enabledFeatures.footholds && isAddFoot ? "#9333ea" : "#f59e0b")}>
            <Edit size={13} />
          </button>
          <button onClick={() => onSetMode("remove")} style={modeBtn(mode === "remove", "#ef4444")}>
            <Eraser size={13} />
          </button>
          <button onClick={() => onSetMode("select")} style={modeBtn(mode === "select", "#3b82f6")}>
            <Hand size={13} />
          </button>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <button
          onClick={onClear}
          className="bz-mono"
          style={{ padding: "6px 12px", borderRadius: "var(--radius)", fontSize: "0.6rem", letterSpacing: "0.08em", textTransform: "uppercase", background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", transition: "all 0.15s" }}
          onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(239,68,68,0.1)"; e.currentTarget.style.color = "#ef4444"; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--text-muted)"; }}
        >CLEAR</button>
        <button
          onClick={onCancel}
          className="bz-mono"
          style={{ padding: "6px 12px", fontSize: "0.6rem", letterSpacing: "0.08em", textTransform: "uppercase", background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", transition: "color 0.15s" }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
        >CANCEL</button>
        <button
          onClick={onSave}
          disabled={isSubmitting}
          className="bz-oswald"
          style={{
            padding: "6px 18px", border: "none", borderRadius: "var(--radius)",
            fontSize: "0.75rem", letterSpacing: "0.1em", fontWeight: 700, textTransform: "uppercase",
            background: isSubmitting ? "var(--surface2)" : "var(--cyan)",
            color: isSubmitting ? "var(--text-muted)" : "#09090b",
            cursor: isSubmitting ? "not-allowed" : "pointer",
            transition: "all 0.15s", opacity: isSubmitting ? 0.6 : 1,
          }}
        >{isSubmitting ? "SAVING…" : "SAVE HOLDS"}</button>
      </div>
    </header>
  );
}

// ── HoldsEditorPage ────────────────────────────────────────────────────────────

function HoldsEditorPage() {
  const navigate = useNavigate();
  const { layout } = Route.useLoaderData() as { layout: LayoutDetail };
  const layoutId = layout.metadata.id;
  const dims = layout.metadata.dimensions;
  const wallDimensions = { width: dims[0] ?? 12, height: dims[1] ?? 12 };

  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [mode, setHoldMode] = useState<HoldMode>("add");
  const [selectedHold, setSelectedHold] = useState<HoldDetail | null>(null);
  const [isAddFoot, setIsAddFoot] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFeatureMenu, setShowFeatureMenu] = useState(false);
  const [enabledFeatures, setEnabledFeatures] = useState<EnabledFeatures>({
    direction: true, useability: true, footholds: true, tags: true,
  });
  const [useabilityLocked, setUseabilityLocked] = useState(false);
  const [lockedUseability, setLockedUseability] = useState(0.5);
  const [activeHoldIndex, setActiveHoldIndex] = useState<number | null>(null);
  const [stickyTag, setStickyTag] = useState<Tag | null>(null);
  const [addDragState, setAddDragState] = useState<AddDragState>({ isDragging: false, holdX: 0, holdY: 0, dragX: 0, dragY: 0 });
  const [editDragState, setEditDragState] = useState<EditDragState>({ isDragging: false, dragX: 0, dragY: 0 });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const panRef = useRef({ isDragging: false, startX: 0, startY: 0, startViewX: 0, startViewY: 0 });

  const imageEdges = layout.metadata.image_edges as [number, number, number, number] | null;
  const homographySrcCorners = layout.metadata.homography_src_corners ?? null;

  const {
    holds, addHold, updateHold, removeHold, removeHoldByIndex,
    removeLastHold, findHoldAt, clearHolds, loadHolds, toPixelCoords, toFeetCoords,
  } = useHolds(imageDimensions, wallDimensions, imageEdges, homographySrcCorners);

  const activeHold = activeHoldIndex !== null
    ? (holds.find((h) => h.hold_index === activeHoldIndex) ?? null)
    : null;

  const { viewTransform, setViewTransform } = useZoomPan(wrapperRef as React.RefObject<HTMLDivElement>);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImage(img);
      setImageDimensions({ width: img.width, height: img.height });
      if (wrapperRef.current) {
        const rect = wrapperRef.current.getBoundingClientRect();
        const scale = Math.min(rect.width / img.width, rect.height / img.height) * 0.9;
        setViewTransform({ zoom: scale, x: (rect.width - img.width * scale) / 2, y: (rect.height - img.height * scale) / 2 });
      }
    };
    img.src = getLayoutPhotoUrl(layoutId);
  }, [layoutId, setViewTransform]);

  // Load holds
  useEffect(() => {
    if (layout.holds && imageDimensions.width > 0) loadHolds(layout.holds);
  }, [layout.holds, loadHolds, imageDimensions.width]);

  const getImageCoords = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.round((e.clientX - rect.left) * (imageDimensions.width / rect.width)),
        y: Math.round((e.clientY - rect.top) * (imageDimensions.height / rect.height)),
      };
    },
    [imageDimensions],
  );

  const calculateHoldParams = useCallback(
    (hx: number, hy: number, dx: number, dy: number): HoldParams => {
      const px = hx - dx, py = hy - dy;
      const mag = Math.sqrt(px * px + py * py);
      const { x, y } = toFeetCoords(hx, hy);
      return {
        pull_x: mag === 0 ? 0 : px / mag,
        pull_y: mag === 0 ? -1 : py / mag,
        useability: enabledFeatures.useability
          ? useabilityLocked ? lockedUseability : Math.min(1, mag / 250)
          : 0.5,
        x, y,
      };
    },
    [toFeetCoords, enabledFeatures, lockedUseability, useabilityLocked],
  );

  const getHoldColor = useCallback((u: number, isFoot: boolean) => {
    if (isFoot)
      return `rgb(${Math.round(60 - 60 * u)}, ${Math.round(200 * u)}, ${Math.round(40 + 140 * u)})`;
    const r = u < 0.5 ? 255 : Math.round(60 + 195 * (1 - u) * 2);
    const g = u < 0.5 ? Math.round(60 + 160 * u * 2) : 220;
    return `rgb(${r}, ${g}, 60)`;
  }, []);

  useHoldHotkeys({ enabledFeatures, removeLastHold, setHoldMode, setIsAddFoot, setStickyTag });

  useCanvasDraw(canvasRef as React.RefObject<HTMLCanvasElement>, {
    image, imageDimensions, holds, mode, selectedHold,
    addDragState, editDragState, enabledFeatures, isAddFoot, stickyTag,
    toPixelCoords, calculateHoldParams, getHoldColor,
  });

  // ── Mouse handlers ─────────────────────────────────────────────────────────

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getImageCoords(e);
    if (e.button === 1 || e.shiftKey) {
      panRef.current = { isDragging: true, startX: e.clientX, startY: e.clientY, startViewX: viewTransform.x, startViewY: viewTransform.y };
      return;
    }
    if (mode === "add") {
      if (enabledFeatures.tags && stickyTag) {
        const existing = findHoldAt(x, y);
        if (existing?.tags.includes(stickyTag)) {
          updateHold(existing.hold_index, { tags: existing.tags.filter((t) => t !== stickyTag) });
          return;
        }
      }
      if (!enabledFeatures.direction && !enabledFeatures.useability) {
        const idx = addHold(x, y, undefined, undefined, undefined, undefined, enabledFeatures.footholds && isAddFoot, enabledFeatures.tags && stickyTag ? [stickyTag] : []);
        setActiveHoldIndex(idx);
      } else {
        setAddDragState({ isDragging: true, holdX: x, holdY: y, dragX: x, dragY: y });
      }
    } else if (mode === "edit") {
      const hold = findHoldAt(x, y);
      if (hold) {
        const pc = toPixelCoords(hold);
        setEditDragState({ isDragging: true, dragX: pc.x, dragY: pc.y, originalHold: { ...hold } });
      }
    } else if (mode === "remove") {
      removeHold(x, y);
    } else if (mode === "select") {
      const hold = findHoldAt(x, y);
      setSelectedHold(hold);
      if (hold) setActiveHoldIndex(hold.hold_index);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (panRef.current.isDragging) {
      setViewTransform((prev) => ({
        ...prev,
        x: panRef.current.startViewX + (e.clientX - panRef.current.startX),
        y: panRef.current.startViewY + (e.clientY - panRef.current.startY),
      }));
    } else if (addDragState.isDragging) {
      const { x, y } = getImageCoords(e);
      setAddDragState((prev) => ({ ...prev, dragX: x, dragY: y }));
    } else if (editDragState.isDragging) {
      const { x, y } = getImageCoords(e);
      setEditDragState((prev) => ({ ...prev, dragX: x, dragY: y }));
    }
  };

  const handleMouseUp = () => {
    if (panRef.current.isDragging) { panRef.current.isDragging = false; return; }
    if (addDragState.isDragging) {
      const { holdX, holdY, dragX, dragY } = addDragState;
      const params = calculateHoldParams(holdX, holdY, dragX, dragY);
      const idx = addHold(
        holdX, holdY, undefined,
        enabledFeatures.direction ? params.pull_x : undefined,
        enabledFeatures.direction ? params.pull_y : undefined,
        enabledFeatures.useability ? params.useability : undefined,
        enabledFeatures.footholds && isAddFoot,
        enabledFeatures.tags && stickyTag ? [stickyTag] : [],
      );
      setActiveHoldIndex(idx);
      setAddDragState({ isDragging: false, holdX: 0, holdY: 0, dragX: 0, dragY: 0 });
    } else if (editDragState.isDragging && editDragState.originalHold) {
      const { originalHold, dragX, dragY } = editDragState;
      const pc = toPixelCoords(originalHold);
      const params = calculateHoldParams(pc.x, pc.y, dragX, dragY);
      const existingTags = (holds.find((h) => h.hold_index === originalHold.hold_index) ?? originalHold).tags ?? [];
      updateHold(originalHold.hold_index, {
        ...(enabledFeatures.direction ? { pull_x: params.pull_x, pull_y: params.pull_y } : {}),
        ...(enabledFeatures.useability ? { useability: params.useability } : {}),
        ...(enabledFeatures.footholds ? { is_foot: isAddFoot } : {}),
        ...(enabledFeatures.tags && stickyTag
          ? { tags: existingTags.includes(stickyTag) ? existingTags.filter((t) => t !== stickyTag) : [...existingTags, stickyTag] }
          : {}),
      });
      setEditDragState({ isDragging: false, dragX: 0, dragY: 0 });
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const dragParams = getDragParams(addDragState, editDragState, toPixelCoords, calculateHoldParams);

  const handleSave = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      await setLayoutHolds(layoutId, holds);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save holds");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <div style={{ position: "relative", height: "calc(100vh - 0px)", display: "flex", flexDirection: "column", background: "var(--bg)", overflow: "hidden", color: "var(--text-primary)" }}>
        {showFeatureMenu && (
          <EnabledFeaturesMenu
            enabledFeatures={enabledFeatures}
            onToggle={(f) => setEnabledFeatures((prev) => ({ ...prev, [f]: !prev[f] }))}
            onClose={() => setShowFeatureMenu(false)}
          />
        )}

        <HoldsHeader
          layoutName={layout.metadata.name}
          mode={mode}
          enabledFeatures={enabledFeatures}
          isAddFoot={isAddFoot}
          showFeatureMenu={showFeatureMenu}
          isSubmitting={isSubmitting}
          onToggleFeatureMenu={() => setShowFeatureMenu((p) => !p)}
          onSetMode={setHoldMode}
          onClear={() => { clearHolds(); setSelectedHold(null); }}
          onCancel={() => navigate({ to: "/$layoutId/set", params: { layoutId } })}
          onSave={handleSave}
        />

        {error && (
          <div className="bz-mono" style={{ padding: "8px 20px", background: "rgba(248,113,113,0.08)", borderBottom: "1px solid rgba(248,113,113,0.2)", fontSize: "0.65rem", color: "#f87171" }}>
            {error}
          </div>
        )}

        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <div ref={wrapperRef} style={{ flex: 1, overflow: "hidden", background: "var(--bg)", cursor: "crosshair" }}>
            <div style={{ transform: `translate(${viewTransform.x}px, ${viewTransform.y}px)` }}>
              <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{
                  width: (imageDimensions.width || 800) * viewTransform.zoom,
                  height: (imageDimensions.height || 600) * viewTransform.zoom,
                }}
              />
            </div>
          </div>

          <HoldFeaturesSidebar
            mode={mode}
            enabledFeatures={enabledFeatures}
            selectedHold={selectedHold}
            isDragging={addDragState.isDragging || editDragState.isDragging}
            dragParams={dragParams}
            getColor={(u) => getHoldColor(u, false)}
            onDeleteHold={() => {
              if (selectedHold) { removeHoldByIndex(selectedHold.hold_index); setSelectedHold(null); }
            }}
            useabilityLocked={useabilityLocked}
            lockedUseability={lockedUseability}
            onUseabilityLockChange={setUseabilityLocked}
            onLockedUseabilityChange={setLockedUseability}
            activeHold={activeHold}
            onTagToggle={(tag) => {
              if (activeHoldIndex === null) return;
              const hold = holds.find((h) => h.hold_index === activeHoldIndex);
              if (!hold) return;
              const newTags = hold.tags.includes(tag) ? hold.tags.filter((t) => t !== tag) : [...hold.tags, tag];
              updateHold(activeHoldIndex, { tags: newTags });
            }}
            stickyTag={stickyTag}
            onStickyTagToggle={(tag) => setStickyTag((p) => (p === tag ? null : tag))}
          />
        </div>
      </div>
    </>
  );
}
