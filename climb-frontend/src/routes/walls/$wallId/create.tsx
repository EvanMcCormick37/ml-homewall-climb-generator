import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { getWall, getWallPhotoUrl } from "@/api/walls";
import { createClimb } from "@/api/climbs";
import { ArrowLeft, Plus, X, ChevronDown } from "lucide-react";
import type { WallDetail, HoldDetail } from "@/types";

// --- Route Definition ---

export const Route = createFileRoute("/walls/$wallId/create")({
  component: CreateClimbPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

// --- Constants ---

// Hold categories for the data model
export type HoldCategory = "hand" | "foot" | "start" | "finish";

const CATEGORY_ORDER: HoldCategory[] = ["hand", "foot", "start", "finish"];

const CATEGORY_COLORS: Record<HoldCategory, string> = {
  hand: "#3b82f6",
  foot: "#a855f7",
  start: "#22c55e",
  finish: "#ffea00ff",
};

const CATEGORY_LABELS: Record<HoldCategory, string> = {
  hand: "Hand",
  foot: "Foot",
  start: "Start",
  finish: "Finish",
};

const HOLD_STROKE_COLOR = "#00b679";

// V-grade options for the dropdown
const GRADE_OPTIONS = [
  { value: null, label: "Project (Ungraded)" },
  ...[
    "V0-",
    "V0",
    "V0+",
    "V1",
    "V1+",
    "V2",
    "V2+",
    "V3",
    "V3+",
    "V4",
    "V4+",
    "V5",
    "V5+",
    "V6",
    "V6+",
    "V7",
    "V7+",
    "V8",
    "V8+",
    "V9",
    "V9+",
    "V10",
    "V10+",
    "V11",
    "V11+",
    "V12",
    "V12+",
    "V13",
    "V13+",
    "V14",
    "V14+",
    "V15",
    "V15+",
    "V16",
  ].map((v) => ({ value: v, label: v })),
];

const ANGLES = [
  -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
  80, 85, 90,
];

// Common climbing tags
const SUGGESTED_TAGS = [
  "crimps",
  "slopers",
  "pinches",
  "jugs",
  "dynamic",
  "static",
  "technical",
  "powerful",
  "overhang",
  "slab",
  "compression",
  "heel-hook",
  "toe-hook",
  "mantle",
  "campus",
];

// --- Types ---

interface HoldSelection {
  holdIndex: number;
  category: HoldCategory;
}

interface ClimbFormData {
  name: string;
  angle: number | null;
  grade: number | null;
  setter_name: string;
  tags: string[];
}

// --- ClimbForm Component ---

interface ClimbFormProps {
  formData: ClimbFormData;
  wallAngle: number | null;
  onFormChange: (data: Partial<ClimbFormData>) => void;
  selectedHolds: HoldSelection[];
  onSubmit: () => void;
  onReset: () => void;
  isSubmitting: boolean;
  error: string | null;
  canSubmit: boolean;
}

function ClimbForm({
  formData,
  wallAngle,
  onFormChange,
  selectedHolds,
  onSubmit,
  onReset,
  isSubmitting,
  error,
  canSubmit,
}: ClimbFormProps) {
  const [isGradeOpen, setIsGradeOpen] = useState(false);
  const [isAngleOpen, setIsAngleOpen] = useState(false);
  const [tagInput, setTagInput] = useState("");

  const selectedGradeLabel =
    GRADE_OPTIONS.find((opt) => opt.value === formData.grade)?.label ||
    "Select Grade";

  // Count holds by category
  const holdCounts = useMemo(() => {
    const counts: Record<HoldCategory, number> = {
      hand: 0,
      foot: 0,
      start: 0,
      finish: 0,
    };
    selectedHolds.forEach((h) => counts[h.category]++);
    return counts;
  }, [selectedHolds]);

  const handleAddTag = (tag: string) => {
    const trimmed = tag.trim().toLowerCase();
    if (trimmed && !formData.tags.includes(trimmed)) {
      onFormChange({ tags: [...formData.tags, trimmed] });
    }
    setTagInput("");
  };

  const handleRemoveTag = (tag: string) => {
    onFormChange({ tags: formData.tags.filter((t) => t !== tag) });
  };

  return (
    <div className="w-80 flex-shrink-0 border-l border-zinc-800 bg-zinc-900 flex flex-col">
      <div className="p-4 border-b border-zinc-800">
        <h2 className="text-lg font-semibold text-zinc-100">Create Climb</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Click holds to select them for your climb
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Name */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-1">
            Name <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => onFormChange({ name: e.target.value })}
            placeholder="Enter climb name"
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                       text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Grade */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-1">
            Grade
          </label>
          <div className="relative">
            <button
              type="button"
              onClick={() => setIsGradeOpen(!isGradeOpen)}
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                         text-zinc-100 text-left flex items-center justify-between"
            >
              {selectedGradeLabel}
              <ChevronDown className="w-4 h-4 text-zinc-500" />
            </button>
            {isGradeOpen && (
              <div
                className="absolute top-full left-0 right-0 mt-1 bg-zinc-800 border border-zinc-700 
                              rounded-lg shadow-lg max-h-48 overflow-y-auto z-10"
              >
                {GRADE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value ?? "null"}
                    type="button"
                    onClick={() => {
                      onFormChange({ grade: opt.value });
                      setIsGradeOpen(false);
                    }}
                    className={`w-full px-3 py-2 text-left hover:bg-zinc-700 transition-colors
                      ${formData.grade === opt.value ? "bg-zinc-700 text-emerald-400" : "text-zinc-300"}`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Angle */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-1">
            Angle
          </label>
          {!!wallAngle ? (
            <div
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                         text-zinc-100 text-left flex items-center justify-between"
            >
              {wallAngle}
            </div>
          ) : (
            <div className="relative">
              <button
                type="button"
                onClick={() => setIsAngleOpen((prev) => !prev)}
                className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                         text-zinc-100 text-left flex items-center justify-between"
              >
                {selectedGradeLabel}
                <ChevronDown className="w-4 h-4 text-zinc-500" />
              </button>
              {isAngleOpen && (
                <div
                  className="absolute top-full left-0 right-0 mt-1 bg-zinc-800 border border-zinc-700 
                              rounded-lg shadow-lg max-h-48 overflow-y-auto z-10"
                >
                  {ANGLES.map((opt) => (
                    <button
                      key={opt ?? "null"}
                      type="button"
                      onClick={() => {
                        onFormChange({ angle: opt });
                        setIsGradeOpen(false);
                      }}
                      className={`w-full px-3 py-2 text-left hover:bg-zinc-700 transition-colors
                      ${formData.angle === opt ? "bg-zinc-700 text-emerald-400" : "text-zinc-300"}`}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Setter */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-1">
            Setter
          </label>
          <input
            type="text"
            value={formData.setter_name}
            onChange={(e) => onFormChange({ setter_name: e.target.value })}
            placeholder="Who set this climb?"
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                       text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Tags */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-1">
            Tags
          </label>
          <div className="flex flex-wrap gap-1 mb-2">
            {formData.tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 px-2 py-1 bg-zinc-700 
                           text-zinc-200 rounded text-xs"
              >
                {tag}
                <button
                  type="button"
                  onClick={() => handleRemoveTag(tag)}
                  className="hover:text-red-400"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
          <input
            type="text"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAddTag(tagInput);
              }
            }}
            placeholder="Add tags..."
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                       text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
          />
          <div className="flex flex-wrap gap-1 mt-2">
            {SUGGESTED_TAGS.filter((t) => !formData.tags.includes(t))
              .slice(0, 6)
              .map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => handleAddTag(tag)}
                  className="px-2 py-0.5 bg-zinc-800 text-zinc-500 text-xs rounded 
                             hover:bg-zinc-700 hover:text-zinc-300"
                >
                  + {tag}
                </button>
              ))}
          </div>
        </div>

        {/* Hold Selection Summary */}
        <div className="pt-4 border-t border-zinc-800">
          <h3 className="text-sm font-medium text-zinc-400 mb-2">
            Selected Holds
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {CATEGORY_ORDER.map((cat) => (
              <div
                key={cat}
                className="flex items-center gap-2 px-3 py-2 bg-zinc-800 rounded-lg"
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS[cat] }}
                />
                <span className="text-sm text-zinc-300">
                  {CATEGORY_LABELS[cat]}
                </span>
                <span className="ml-auto text-sm font-medium text-zinc-100">
                  {holdCounts[cat]}
                </span>
              </div>
            ))}
          </div>
          <p className="text-xs text-zinc-500 mt-2">
            Click holds to cycle through: Hand → Foot → Start → Finish → Remove
          </p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-2 bg-red-900/50 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Actions */}
      <div className="p-4 border-t border-zinc-800 space-y-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit}
          className={`w-full px-4 py-2.5 rounded-lg font-medium transition-colors flex items-center justify-center gap-2
            ${
              canSubmit
                ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                : "bg-zinc-700 text-zinc-500 cursor-not-allowed"
            }`}
        >
          {isSubmitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Plus className="w-4 h-4" />
              Create Climb
            </>
          )}
        </button>
        <button
          type="button"
          onClick={onReset}
          disabled={isSubmitting}
          className="w-full px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 
                     font-medium rounded-lg transition-colors"
        >
          Reset Form
        </button>
      </div>
    </div>
  );
}

// --- Canvas Component ---

interface CanvasProps {
  wallId: string;
  holds: HoldDetail[];
  wallDimensions: { width: number; height: number };
  selectedHolds: HoldSelection[];
  onHoldClick: (holdIndex: number) => void;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dimensions: { width: number; height: number }) => void;
}

function Canvas({
  wallId,
  holds,
  wallDimensions,
  selectedHolds,
  onHoldClick,
  imageDimensions,
  onImageLoad,
}: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [viewTransform, setViewTransform] = useState({ zoom: 1, x: 0, y: 0 });

  // Pan drag state
  const panDragRef = useRef({
    isDragging: false,
    startX: 0,
    startY: 0,
    startViewX: 0,
    startViewY: 0,
  });

  // Create lookup map for selected holds
  const selectedHoldsMap = useMemo(() => {
    const map = new Map<number, HoldCategory>();
    selectedHolds.forEach((h) => map.set(h.holdIndex, h.category));
    return map;
  }, [selectedHolds]);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      setImage(img);
      onImageLoad({ width: img.width, height: img.height });

      // Fit to container
      if (wrapperRef.current) {
        const rect = wrapperRef.current.getBoundingClientRect();
        const scale =
          Math.min(rect.width / img.width, rect.height / img.height) * 0.95;
        setViewTransform({
          zoom: scale,
          x: (rect.width - img.width * scale) / 2,
          y: (rect.height - img.height * scale) / 2,
        });
      }
    };
    img.src = getWallPhotoUrl(wallId);
  }, [wallId, onImageLoad]);

  // Convert feet to pixel coordinates
  const toPixelCoords = useCallback(
    (hold: HoldDetail): { x: number; y: number } => {
      const { width: imgW, height: imgH } = imageDimensions;
      const { width: wallW, height: wallH } = wallDimensions;
      return {
        x: (hold.x / wallW) * imgW,
        y: (1 - hold.y / wallH) * imgH,
      };
    },
    [imageDimensions, wallDimensions],
  );

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = imageDimensions;
    canvas.width = width || 800;
    canvas.height = height || 600;

    // Clear
    ctx.fillStyle = "#18181b";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(image, 0, 0);

    // Draw holds
    holds.forEach((hold) => {
      const { x, y } = toPixelCoords(hold);
      const size = height / 1800;
      const radius = hold.is_foot ? 10 * size : 18 * size;
      const category = selectedHoldsMap.get(hold.hold_index);
      const isSelected = category !== undefined;

      // Draw selection ring if selected and is not a foot.
      if (isSelected && !hold.is_foot) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, 2 * Math.PI);
        ctx.strokeStyle = CATEGORY_COLORS[category];
        ctx.lineWidth = 4;
        ctx.stroke();
      }

      // Draw hold circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = isSelected
        ? CATEGORY_COLORS[category]
        : HOLD_STROKE_COLOR;
      if (isSelected) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fill();
      }
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw hold index
      ctx.fillStyle = isSelected ? CATEGORY_COLORS[category] : "white";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(hold.hold_index.toString(), x, y);
    });
  }, [image, imageDimensions, holds, selectedHoldsMap, toPixelCoords]);

  // Get image coordinates from mouse event
  const getImageCoords = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) * (imageDimensions.width / rect.width),
        y: (e.clientY - rect.top) * (imageDimensions.height / rect.height),
      };
    },
    [imageDimensions],
  );

  // Find hold at position
  const findHoldAt = useCallback(
    (pixelX: number, pixelY: number): HoldDetail | null => {
      const radius = 25;
      for (const hold of holds) {
        const { x, y } = toPixelCoords(hold);
        const dist = Math.sqrt((x - pixelX) ** 2 + (y - pixelY) ** 2);
        if (dist < radius) return hold;
      }
      return null;
    },
    [holds, toPixelCoords],
  );

  // Handle click
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (panDragRef.current.isDragging) return;
      const { x, y } = getImageCoords(e);
      const hold = findHoldAt(x, y);
      if (hold) {
        onHoldClick(hold.hold_index);
      }
    },
    [getImageCoords, findHoldAt, onHoldClick],
  );

  // Pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.shiftKey || e.button === 1) {
        panDragRef.current = {
          isDragging: true,
          startX: e.clientX,
          startY: e.clientY,
          startViewX: viewTransform.x,
          startViewY: viewTransform.y,
        };
      }
    },
    [viewTransform],
  );

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (panDragRef.current.isDragging) {
      setViewTransform((prev) => ({
        ...prev,
        x:
          panDragRef.current.startViewX +
          (e.clientX - panDragRef.current.startX),
        y:
          panDragRef.current.startViewY +
          (e.clientY - panDragRef.current.startY),
      }));
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    panDragRef.current.isDragging = false;
  }, []);

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const rect = element.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      setViewTransform((prev) => {
        const newZoom = Math.max(0.1, Math.min(10, prev.zoom * zoomFactor));
        const scale = newZoom / prev.zoom;
        return {
          zoom: newZoom,
          x: mouseX - (mouseX - prev.x) * scale,
          y: mouseY - (mouseY - prev.y) * scale,
        };
      });
    };

    element.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      element.removeEventListener("wheel", handleWheel);
    };
  }, []);

  return (
    <div ref={wrapperRef} className="flex-1 overflow-hidden bg-zinc-950">
      <div
        style={{
          transform: `translate(${viewTransform.x}px, ${viewTransform.y}px)`,
        }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="cursor-pointer"
          style={{
            width: (imageDimensions.width || 800) * viewTransform.zoom,
            height: (imageDimensions.height || 600) * viewTransform.zoom,
          }}
        />
      </div>
    </div>
  );
}

// --- Main Page Component ---

function CreateClimbPage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const wallId = wall.metadata.id;
  const wallDimensions = {
    width: wall.metadata.dimensions[0],
    height: wall.metadata.dimensions[1],
  };
  const wallAngle = wall.metadata.angle ?? null;

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [selectedHolds, setSelectedHolds] = useState<HoldSelection[]>([]);
  const [formData, setFormData] = useState<ClimbFormData>({
    name: "",
    angle: wallAngle,
    grade: null,
    setter_name: "",
    tags: [],
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle image load
  const handleImageLoad = useCallback(
    (dimensions: { width: number; height: number }) => {
      setImageDimensions(dimensions);
    },
    [],
  );

  // Handle form reset
  const handleReset = useCallback(() => {
    setSelectedHolds([]);
    setFormData({
      name: "",
      angle: wallAngle,
      grade: null,
      setter_name: "",
      tags: [],
    });
    setError(null);
  }, []);

  // Handle form change
  const handleFormChange = useCallback((updates: Partial<ClimbFormData>) => {
    setFormData((prev) => ({ ...prev, ...updates }));
  }, []);

  // Handle hold click - cycle through categories
  const handleHoldClick = useCallback((holdIndex: number) => {
    setSelectedHolds((prev) => {
      const existing = prev.find((h) => h.holdIndex === holdIndex);

      if (!existing) {
        // Not selected - add as "hand"
        return [...prev, { holdIndex, category: "hand" as HoldCategory }];
      }

      // Get current category index
      const currentIndex = CATEGORY_ORDER.indexOf(existing.category);
      const nextIndex = (currentIndex + 1) % (CATEGORY_ORDER.length + 1);

      // If we've cycled through all, remove the hold
      if (nextIndex === CATEGORY_ORDER.length) {
        return prev.filter((h) => h.holdIndex !== holdIndex);
      }

      const nextCategory = CATEGORY_ORDER[nextIndex];

      // Check limits for start/finish
      if (nextCategory === "start") {
        const startCount = prev.filter((h) => h.category === "start").length;
        if (startCount >= 2 && existing.category !== "start") {
          // Skip to finish if start is full
          const finishCount = prev.filter(
            (h) => h.category === "finish",
          ).length;
          if (finishCount >= 2) {
            // Both full, remove the hold
            return prev.filter((h) => h.holdIndex !== holdIndex);
          }
          return prev.map((h) =>
            h.holdIndex === holdIndex
              ? { ...h, category: "finish" as HoldCategory }
              : h,
          );
        }
      }

      if (nextCategory === "finish") {
        const finishCount = prev.filter((h) => h.category === "finish").length;
        if (finishCount >= 2 && existing.category !== "finish") {
          // Finish is full, remove the hold
          return prev.filter((h) => h.holdIndex !== holdIndex);
        }
      }

      // Update to next category
      return prev.map((h) =>
        h.holdIndex === holdIndex ? { ...h, category: nextCategory } : h,
      );
    });
  }, []);

  // Check if form can be submitted
  const canSubmit = useMemo(() => {
    const hasName = formData.name.trim().length > 0;
    const hasAngle = !!formData.angle;
    const hasStart = selectedHolds.some((h) => h.category === "start");
    const hasFinish = selectedHolds.some((h) => h.category === "finish");
    return hasName && hasAngle && hasStart && hasFinish && !isSubmitting;
  }, [formData.name, selectedHolds, isSubmitting]);

  // Handle form submission
  const handleSubmit = useCallback(async () => {
    if (!canSubmit) return;

    setIsSubmitting(true);
    setError(null);

    try {
      // Build the holds data with the Holdset structure
      const holdset = {
        start: selectedHolds
          .filter((h) => h.category === "start")
          .map((h) => h.holdIndex),
        finish: selectedHolds
          .filter((h) => h.category === "finish")
          .map((h) => h.holdIndex),
        hand: selectedHolds
          .filter((h) => h.category === "hand")
          .map((h) => h.holdIndex),
        foot: selectedHolds
          .filter((h) => h.category === "foot")
          .map((h) => h.holdIndex),
      };

      await createClimb(wallId, {
        name: formData.name.trim(),
        angle: formData.angle!,
        grade: formData.grade,
        setter_name: formData.setter_name.trim() || null,
        tags: formData.tags.length > 0 ? formData.tags : null,
        holdset,
      });

      // Navigate back to wall view after successful submission
      navigate({ to: "/walls/$wallId/view", params: { wallId } });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to create climb";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [canSubmit, wallId, formData, selectedHolds, navigate]);

  return (
    <div className="h-screen flex flex-col bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() =>
              navigate({ to: "/walls/$wallId", params: { wallId } })
            }
            className="flex items-center gap-1 text-zinc-400 hover:text-zinc-100 transition-colors text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <div className="w-px h-5 bg-zinc-700" />
          <h1 className="text-lg font-medium text-zinc-100">
            {wall.metadata.name}
          </h1>
          <span className="text-sm text-zinc-500">Create Climb</span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Canvas */}
        <Canvas
          wallId={wallId}
          holds={wall.holds}
          wallDimensions={wallDimensions}
          selectedHolds={selectedHolds}
          onHoldClick={handleHoldClick}
          imageDimensions={imageDimensions}
          onImageLoad={handleImageLoad}
        />

        {/* Form sidebar */}
        <ClimbForm
          formData={formData}
          wallAngle={wallAngle}
          onFormChange={handleFormChange}
          selectedHolds={selectedHolds}
          onSubmit={handleSubmit}
          onReset={handleReset}
          isSubmitting={isSubmitting}
          error={error}
          canSubmit={canSubmit}
        />
      </div>
    </div>
  );
}
