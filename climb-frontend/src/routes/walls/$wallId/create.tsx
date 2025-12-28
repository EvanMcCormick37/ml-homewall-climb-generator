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

// Hold categories for the new refactored data model
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
  ...Array.from({ length: 18 }, (_, i) => ({
    value: i * 10,
    label: `V${i}`,
  })),
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
  holdId: number;
  category: HoldCategory;
}

interface ClimbFormData {
  name: string;
  grade: number | null;
  setter: string;
  tags: string[];
}

// --- ClimbForm Component ---

interface ClimbFormProps {
  formData: ClimbFormData;
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
  onFormChange,
  selectedHolds,
  onSubmit,
  onReset,
  isSubmitting,
  error,
  canSubmit,
}: ClimbFormProps) {
  const [tagInput, setTagInput] = useState("");
  const [isGradeOpen, setIsGradeOpen] = useState(false);

  const startHolds = selectedHolds.filter((h) => h.category === "start");
  const finishHolds = selectedHolds.filter((h) => h.category === "finish");
  const handHolds = selectedHolds.filter((h) => h.category === "hand");
  const footHolds = selectedHolds.filter((h) => h.category === "foot");

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

  const handleTagKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      handleAddTag(tagInput);
    }
  };

  const gradeLabel =
    formData.grade !== null
      ? `V${Math.floor(formData.grade / 10)}`
      : "Project (Ungraded)";

  // Validation messages
  const validationMessages: string[] = [];
  if (!formData.name.trim()) {
    validationMessages.push("Name required");
  }
  if (startHolds.length === 0) {
    validationMessages.push("At least 1 start hold required");
  }
  if (finishHolds.length === 0) {
    validationMessages.push("At least 1 finish hold required");
  }

  return (
    <div className="flex flex-col h-full bg-zinc-900 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <h2 className="text-lg font-medium text-zinc-100">Create Climb</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Fill in details and select holds on the wall
        </p>
      </div>

      {/* Scrollable Form Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Name Field */}
        <div>
          <label
            htmlFor="climb-name"
            className="block text-sm font-medium text-zinc-400 mb-2"
          >
            Climb Name <span className="text-red-400">*</span>
          </label>
          <input
            id="climb-name"
            type="text"
            value={formData.name}
            onChange={(e) => onFormChange({ name: e.target.value })}
            placeholder="Enter climb name..."
            className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg 
                       text-zinc-100 placeholder-zinc-500 
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none
                       transition-colors"
          />
        </div>

        {/* Grade Field */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-2">
            Grade
          </label>
          <div className="relative">
            <button
              type="button"
              onClick={() => setIsGradeOpen(!isGradeOpen)}
              className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg 
                         text-left text-zinc-100
                         focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none
                         transition-colors flex items-center justify-between"
            >
              <span>{gradeLabel}</span>
              <ChevronDown
                className={`w-4 h-4 text-zinc-500 transition-transform ${
                  isGradeOpen ? "rotate-180" : ""
                }`}
              />
            </button>
            {isGradeOpen && (
              <div
                className="absolute z-10 mt-1 w-full max-h-48 overflow-y-auto 
                              bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl"
              >
                {GRADE_OPTIONS.map((option) => (
                  <button
                    key={option.value ?? "null"}
                    type="button"
                    onClick={() => {
                      onFormChange({ grade: option.value });
                      setIsGradeOpen(false);
                    }}
                    className={`w-full px-3 py-2 text-left text-sm transition-colors
                      ${
                        formData.grade === option.value
                          ? "bg-blue-600 text-white"
                          : "text-zinc-300 hover:bg-zinc-700"
                      }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Setter Field */}
        <div>
          <label
            htmlFor="climb-setter"
            className="block text-sm font-medium text-zinc-400 mb-2"
          >
            Setter
          </label>
          <input
            id="climb-setter"
            type="text"
            value={formData.setter}
            onChange={(e) => onFormChange({ setter: e.target.value })}
            placeholder="Your name (optional)"
            className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg 
                       text-zinc-100 placeholder-zinc-500 
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none
                       transition-colors"
          />
        </div>

        {/* Tags Field */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-2">
            Tags
          </label>

          {/* Selected Tags */}
          {formData.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-2">
              {formData.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs 
                             bg-blue-600/20 text-blue-400 rounded-md"
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => handleRemoveTag(tag)}
                    className="hover:text-blue-200 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}

          {/* Tag Input */}
          <input
            type="text"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyDown={handleTagKeyDown}
            onBlur={() => tagInput && handleAddTag(tagInput)}
            placeholder="Add tags..."
            className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg 
                       text-zinc-100 placeholder-zinc-500 
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none
                       transition-colors"
          />

          {/* Suggested Tags */}
          <div className="flex flex-wrap gap-1 mt-2">
            {SUGGESTED_TAGS.filter((t) => !formData.tags.includes(t))
              .slice(0, 8)
              .map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => handleAddTag(tag)}
                  className="px-2 py-0.5 text-xs text-zinc-500 hover:text-zinc-300 
                             bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
                >
                  + {tag}
                </button>
              ))}
          </div>
        </div>

        {/* Hold Selection Summary */}
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-2">
            Selected Holds
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div className="px-3 py-2 bg-zinc-800 rounded-lg border border-zinc-700">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS.start }}
                />
                <span className="text-sm text-zinc-300">
                  Start: {startHolds.length}/2
                </span>
              </div>
            </div>
            <div className="px-3 py-2 bg-zinc-800 rounded-lg border border-zinc-700">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS.finish }}
                />
                <span className="text-sm text-zinc-300">
                  Finish: {finishHolds.length}/2
                </span>
              </div>
            </div>
            <div className="px-3 py-2 bg-zinc-800 rounded-lg border border-zinc-700">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS.hand }}
                />
                <span className="text-sm text-zinc-300">
                  Hand: {handHolds.length}
                </span>
              </div>
            </div>
            <div className="px-3 py-2 bg-zinc-800 rounded-lg border border-zinc-700">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS.foot }}
                />
                <span className="text-sm text-zinc-300">
                  Foot: {footHolds.length}
                </span>
              </div>
            </div>
          </div>
          <p className="text-xs text-zinc-500 mt-2">
            Click holds on the wall to cycle: Hand → Foot → Start → Finish
          </p>
        </div>

        {/* Validation Messages */}
        {validationMessages.length > 0 && (
          <div className="p-3 bg-yellow-900/20 border border-yellow-700/50 rounded-lg">
            <ul className="text-xs text-yellow-400 space-y-1">
              {validationMessages.map((msg, i) => (
                <li key={i}>• {msg}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-900/20 border border-red-700/50 rounded-lg">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t border-zinc-800 flex-shrink-0 space-y-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit || isSubmitting}
          className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 
                     disabled:text-zinc-500 text-white font-medium rounded-lg 
                     transition-colors flex items-center justify-center gap-2"
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
  selectedHolds: HoldSelection[];
  onHoldClick: (holdId: number) => void;
  imageDimensions: { width: number; height: number };
  onImageLoad: (dimensions: { width: number; height: number }) => void;
}

function Canvas({
  wallId,
  holds,
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
    selectedHolds.forEach((h) => map.set(h.holdId, h.category));
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

  // Convert normalized hold coords to pixel coords
  const getHoldPixelCoords = useCallback(
    (hold: HoldDetail): { x: number; y: number } => {
      return {
        x: hold.norm_x * imageDimensions.width,
        y: (1 - hold.norm_y) * imageDimensions.height,
      };
    },
    [imageDimensions]
  );

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = imageDimensions;
    canvas.width = width;
    canvas.height = height;

    // Clear and draw image
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    // Draw all holds
    holds.forEach((hold) => {
      const { x, y } = getHoldPixelCoords(hold);
      const radius = 18;
      const isSelected = selectedHoldsMap.has(hold.hold_id);
      const category = selectedHoldsMap.get(hold.hold_id);

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);

      if (isSelected && category) {
        // Draw filled circle with category color
        ctx.fillStyle = CATEGORY_COLORS[category];
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = CATEGORY_COLORS[category];
        ctx.lineWidth = 3;
        ctx.stroke();

        // Draw category label
        ctx.font = "bold 11px system-ui";
        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(CATEGORY_LABELS[category][0], x, y);
      } else {
        // Draw default hold outline
        ctx.strokeStyle = HOLD_STROKE_COLOR;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw hold ID
      ctx.font = "10px system-ui";
      ctx.fillStyle = isSelected ? "#fff" : "#888";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(String(hold.hold_id), x, y + radius + 2);
    });
  }, [image, holds, imageDimensions, selectedHoldsMap, getHoldPixelCoords]);

  // Find hold at position
  const findHoldAt = useCallback(
    (clientX: number, clientY: number): HoldDetail | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;

      const rect = canvas.getBoundingClientRect();
      const { zoom, x: viewX, y: viewY } = viewTransform;

      // Convert client coords to canvas coords
      const canvasX = (clientX - rect.left - viewX) / zoom;
      const canvasY = (clientY - rect.top - viewY) / zoom;

      // Find closest hold within radius
      const radius = 25;
      let closest: HoldDetail | null = null;
      let minDist = Infinity;

      holds.forEach((hold) => {
        const { x, y } = getHoldPixelCoords(hold);
        const dist = Math.sqrt((x - canvasX) ** 2 + (y - canvasY) ** 2);
        if (dist < radius && dist < minDist) {
          minDist = dist;
          closest = hold;
        }
      });

      return closest;
    },
    [holds, viewTransform, getHoldPixelCoords]
  );

  // Mouse handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Check if clicking on a hold
      const hold = findHoldAt(e.clientX, e.clientY);
      if (hold) {
        onHoldClick(hold.hold_id);
        return;
      }

      // Otherwise, start panning
      panDragRef.current = {
        isDragging: true,
        startX: e.clientX,
        startY: e.clientY,
        startViewX: viewTransform.x,
        startViewY: viewTransform.y,
      };
    },
    [findHoldAt, onHoldClick, viewTransform]
  );

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!panDragRef.current.isDragging) return;

    const dx = e.clientX - panDragRef.current.startX;
    const dy = e.clientY - panDragRef.current.startY;

    setViewTransform((prev) => ({
      ...prev,
      x: panDragRef.current.startViewX + dx,
      y: panDragRef.current.startViewY + dy,
    }));
  }, []);

  const handleMouseUp = useCallback(() => {
    panDragRef.current.isDragging = false;
  }, []);

  // Wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;

    setViewTransform((prev) => {
      const newZoom = Math.max(0.1, Math.min(5, prev.zoom * delta));
      const rect = e.currentTarget.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Zoom towards cursor
      const beforeX = (mouseX - prev.x) / prev.zoom;
      const beforeY = (mouseY - prev.y) / prev.zoom;

      return {
        zoom: newZoom,
        x: mouseX - beforeX * newZoom,
        y: mouseY - beforeY * newZoom,
      };
    });
  }, []);

  const { zoom, x, y } = viewTransform;
  const { width, height } = imageDimensions;

  return (
    <div
      ref={wrapperRef}
      className="w-full h-full overflow-hidden bg-zinc-950 cursor-grab active:cursor-grabbing relative"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* Legend */}
      <div className="absolute top-4 right-4 bg-zinc-900/90 backdrop-blur-sm border border-zinc-700 rounded-lg p-3 z-10">
        <p className="text-xs text-zinc-400 mb-2 font-medium">
          Hold Categories
        </p>
        <div className="space-y-1.5">
          {CATEGORY_ORDER.map((category) => (
            <div key={category} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: CATEGORY_COLORS[category] }}
              />
              <span className="text-xs text-zinc-300">
                {CATEGORY_LABELS[category]}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          transform: `translate(${x}px, ${y}px)`,
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: (width || 800) * zoom,
            height: (height || 600) * zoom,
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

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });

  // Form state
  const [formData, setFormData] = useState<ClimbFormData>({
    name: "",
    grade: null,
    setter: "",
    tags: [],
  });

  // Selected holds state
  const [selectedHolds, setSelectedHolds] = useState<HoldSelection[]>([]);

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageLoad = useCallback(
    (dimensions: { width: number; height: number }) => {
      setImageDimensions(dimensions);
    },
    []
  );

  const handleFormChange = useCallback((updates: Partial<ClimbFormData>) => {
    setFormData((prev) => ({ ...prev, ...updates }));
  }, []);

  // Handle hold click - cycle through categories
  const handleHoldClick = useCallback((holdId: number) => {
    setSelectedHolds((prev) => {
      const existing = prev.find((h) => h.holdId === holdId);

      if (!existing) {
        // Not selected - add as "hand"
        return [...prev, { holdId, category: "hand" as HoldCategory }];
      }

      // Get current category index
      const currentIndex = CATEGORY_ORDER.indexOf(existing.category);
      const nextIndex = (currentIndex + 1) % (CATEGORY_ORDER.length + 1);

      // If we've cycled through all, remove the hold
      if (nextIndex === CATEGORY_ORDER.length) {
        return prev.filter((h) => h.holdId !== holdId);
      }

      const nextCategory = CATEGORY_ORDER[nextIndex];

      // Check limits for start/finish
      if (nextCategory === "start") {
        const startCount = prev.filter((h) => h.category === "start").length;
        if (startCount >= 2 && existing.category !== "start") {
          // Skip to finish if start is full
          const finishCount = prev.filter(
            (h) => h.category === "finish"
          ).length;
          if (finishCount >= 2) {
            // Both full, remove the hold
            return prev.filter((h) => h.holdId !== holdId);
          }
          return prev.map((h) =>
            h.holdId === holdId
              ? { ...h, category: "finish" as HoldCategory }
              : h
          );
        }
      }

      if (nextCategory === "finish") {
        const finishCount = prev.filter((h) => h.category === "finish").length;
        if (finishCount >= 2 && existing.category !== "finish") {
          // Finish is full, remove the hold
          return prev.filter((h) => h.holdId !== holdId);
        }
      }

      // Update to next category
      return prev.map((h) =>
        h.holdId === holdId ? { ...h, category: nextCategory } : h
      );
    });
  }, []);

  // Check if form can be submitted
  const canSubmit = useMemo(() => {
    const hasName = formData.name.trim().length > 0;
    const hasStart = selectedHolds.some((h) => h.category === "start");
    const hasFinish = selectedHolds.some((h) => h.category === "finish");
    return hasName && hasStart && hasFinish && !isSubmitting;
  }, [formData.name, selectedHolds, isSubmitting]);

  // Handle form submission
  const handleSubmit = useCallback(async () => {
    if (!canSubmit) return;

    setIsSubmitting(true);
    setError(null);

    try {
      // Build the holds data with the new format
      const holds = {
        start: selectedHolds
          .filter((h) => h.category === "start")
          .map((h) => h.holdId),
        finish: selectedHolds
          .filter((h) => h.category === "finish")
          .map((h) => h.holdId),
        hand: selectedHolds
          .filter((h) => h.category === "hand")
          .map((h) => h.holdId),
        foot: selectedHolds
          .filter((h) => h.category === "foot")
          .map((h) => h.holdId),
      };

      await createClimb(wallId, {
        name: formData.name.trim(),
        grade: formData.grade,
        setter: formData.setter.trim() || null,
        tags: formData.tags.length > 0 ? formData.tags : null,
        holds,
      });

      // Reset form after successful submission
      handleReset();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to create climb";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [canSubmit, wallId, formData, selectedHolds]);

  // Handle form reset
  const handleReset = useCallback(() => {
    setFormData({
      name: "",
      grade: null,
      setter: "",
      tags: [],
    });
    setSelectedHolds([]);
    setError(null);
  }, []);

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
        {/* Left panel - ClimbForm */}
        <div className="w-80 flex-shrink-0 border-r border-zinc-800">
          <ClimbForm
            formData={formData}
            onFormChange={handleFormChange}
            selectedHolds={selectedHolds}
            onSubmit={handleSubmit}
            onReset={handleReset}
            isSubmitting={isSubmitting}
            error={error}
            canSubmit={canSubmit}
          />
        </div>

        {/* Right panel - Canvas */}
        <div className="flex-1 min-w-0">
          <Canvas
            wallId={wallId}
            holds={wall.holds}
            selectedHolds={selectedHolds}
            onHoldClick={handleHoldClick}
            imageDimensions={imageDimensions}
            onImageLoad={handleImageLoad}
          />
        </div>
      </div>
    </div>
  );
}
