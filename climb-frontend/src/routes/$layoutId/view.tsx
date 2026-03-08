import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useEffect } from "react";
import { useLayout, useClimbs } from "@/hooks";
import { useBetaZeroAuth } from "@/hooks/useAuth";
import { WakingScreen } from "@/components";
import { GLOBAL_STYLES } from "@/styles";
import {
  ArrowLeft,
  Loader2,
  ChartNetwork,
  User,
  Calendar,
  Hash,
  Tag,
  Search,
  SunMedium,
  ChevronDown,
  X,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import {
  type Climb,
  type LayoutDetail,
  type GradeScale,
  type ClimbFilters,
  DEFAULT_CLIMB_FILTERS,
  type ClimbSortBy,
} from "@/types";
import { gradeToString, gradeToColor } from "@/utils/climbs";
import {
  VGRADE_OPTIONS,
  FONT_OPTIONS,
  DEFAULT_DISPLAY_SETTINGS,
  type DisplaySettings,
  TogglePair,
  SectionLabel,
  MobileSwipeNav,
  WallCanvas,
  DisplaySettingsPanel,
} from "@/components/wall";

// ─── Route ───────────────────────────────────────────────────────────────────

export const Route = createFileRoute("/$layoutId/view")({
  component: ViewPage,
  staleTime: 3_600_000,
});

// ─── Grade helpers ────────────────────────────────────────────────────────────

const SORT_OPTIONS = [
  { label: "Date Added", value: "date" },
  { label: "Ascents", value: "ascents" },
  { label: "Grade", value: "grade" },
  { label: "Quality", value: "quality" },
];

// ─── FilterPanel ─────────────────────────────────────────────────────────────

const selectStyle: React.CSSProperties = {
  width: "100%",
  background: "var(--bg)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  color: "var(--text-primary)",
  fontFamily: "'Space Mono', monospace",
  fontSize: "0.6rem",
  padding: "5px 8px",
  outline: "none",
  cursor: "pointer",
  appearance: "none",
  WebkitAppearance: "none",
};

const labelStyle: React.CSSProperties = {
  fontFamily: "'Space Mono', monospace",
  fontSize: "0.55rem",
  color: "var(--text-dim)",
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  marginBottom: "4px",
  display: "block",
};

function FilterPanel({
  filters,
  onChange,
  onReset,
}: {
  filters: ClimbFilters;
  onChange: (f: ClimbFilters) => void;
  onReset: () => void;
}) {
  const set = <K extends keyof ClimbFilters>(key: K, val: ClimbFilters[K]) =>
    onChange({ ...filters, [key]: val });
  const [gradeOptions, setGradeOptions] = useState(VGRADE_OPTIONS);
  const changeGradeScale = useCallback((scale: GradeScale) => {
    if (scale === "v_grade") {
      set("gradeScale", "v_grade");
      setGradeOptions(VGRADE_OPTIONS);
      set("minGrade", "V0-");
      set("maxGrade", "V16");
    } else {
      set("gradeScale", "font");
      setGradeOptions(FONT_OPTIONS);
      set("minGrade", "4a");
      set("maxGrade", "8c+");
    }
  }, []);

  const hasChanges =
    filters.minGrade !== DEFAULT_CLIMB_FILTERS.minGrade ||
    filters.maxGrade !== DEFAULT_CLIMB_FILTERS.maxGrade ||
    !filters.includeProjects ||
    filters.setterName !== "" ||
    filters.after !== "" ||
    filters.sortBy !== DEFAULT_CLIMB_FILTERS.sortBy ||
    filters.descending !== DEFAULT_CLIMB_FILTERS.descending;

  return (
    <div
      style={{
        padding: "10px 12px",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        background: "var(--bg)",
        animation: "bzFadeUp 0.15s ease-out",
      }}
    >
      {/* Sort row */}
      <div style={{ display: "flex", gap: "8px" }}>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Sort By</label>
          <div style={{ position: "relative" }}>
            <select
              value={filters.sortBy}
              onChange={(e) => set("sortBy", e.target.value as ClimbSortBy)}
              style={selectStyle}
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
            <ChevronDown
              size={10}
              style={{
                position: "absolute",
                right: "8px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--text-dim)",
                pointerEvents: "none",
              }}
            />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Direction</label>
          <div style={{ display: "flex", gap: "4px" }}>
            {[
              { label: "↓ Desc", val: true },
              { label: "↑ Asc", val: false },
            ].map(({ label, val }) => (
              <button
                key={label}
                onClick={() => set("descending", val)}
                style={{
                  flex: 1,
                  padding: "5px 0",
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.55rem",
                  border: `1px solid ${filters.descending === val ? "var(--cyan)" : "var(--border)"}`,
                  background:
                    filters.descending === val
                      ? "var(--cyan-dim)"
                      : "transparent",
                  color:
                    filters.descending === val
                      ? "var(--cyan)"
                      : "var(--text-muted)",
                  borderRadius: "var(--radius)",
                  cursor: "pointer",
                  transition: "all 0.12s",
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grade range */}
      <div>
        <label style={labelStyle}>Grade Range</label>
        <TogglePair
          options={[
            { value: "v_grade", label: "V-grade" },
            { value: "font", label: "Fontainebleau" },
          ]}
          value={filters.gradeScale}
          onChange={(v) => changeGradeScale(v as GradeScale)}
        />
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <div style={{ flex: 1, position: "relative" }}>
            <select
              value={filters.minGrade}
              onChange={(e) => {
                set("minGrade", e.target.value);
              }}
              style={selectStyle}
            >
              {gradeOptions.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
            <ChevronDown
              size={10}
              style={{
                position: "absolute",
                right: "8px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--text-dim)",
                pointerEvents: "none",
              }}
            />
          </div>
          <span
            style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.55rem",
              color: "var(--text-dim)",
            }}
          >
            to
          </span>
          <div style={{ flex: 1, position: "relative" }}>
            <select
              value={filters.maxGrade}
              onChange={(e) => {
                set("maxGrade", e.target.value);
              }}
              style={selectStyle}
            >
              {gradeOptions.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
            <ChevronDown
              size={10}
              style={{
                position: "absolute",
                right: "8px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--text-dim)",
                pointerEvents: "none",
              }}
            />
          </div>
        </div>
      </div>

      {/* Include projects toggle */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span
          style={{
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.6rem",
            color: "var(--text-muted)",
          }}
        >
          Include ungraded
        </span>
        <button
          onClick={() => set("includeProjects", !filters.includeProjects)}
          style={{
            width: "36px",
            height: "18px",
            borderRadius: "9px",
            border: "none",
            cursor: "pointer",
            position: "relative",
            background: filters.includeProjects
              ? "var(--cyan)"
              : "var(--border)",
            transition: "background 0.15s",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              background: "#fff",
              position: "absolute",
              top: "3px",
              left: filters.includeProjects ? "21px" : "3px",
              transition: "left 0.15s",
            }}
          />
        </button>
      </div>

      {/* Setter name */}
      <div>
        <label style={labelStyle}>Setter</label>
        <input
          type="text"
          placeholder="Filter by setter name…"
          value={filters.setterName}
          onChange={(e) => set("setterName", e.target.value)}
          style={{
            ...selectStyle,
            padding: "5px 8px",
          }}
        />
      </div>

      {/* After date */}
      <div>
        <label style={labelStyle}>Added After</label>
        <input
          type="date"
          value={filters.after}
          onChange={(e) => set("after", e.target.value)}
          style={{
            ...selectStyle,
            padding: "4px 8px",
            colorScheme: "dark",
          }}
        />
      </div>

      {/* Reset */}
      {hasChanges && (
        <button
          onClick={onReset}
          style={{
            alignSelf: "flex-start",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            color: "var(--text-dim)",
            fontFamily: "'Space Mono', monospace",
            fontSize: "0.55rem",
            padding: "4px 10px",
            cursor: "pointer",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          Reset filters
        </button>
      )}
    </div>
  );
}

// ─── DeleteConfirmModal ───────────────────────────────────────────────────────

function DeleteConfirmModal({
  climb,
  onConfirm,
  onCancel,
}: {
  climb: Climb;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(0,0,0,0.7)",
        }}
        onClick={onCancel}
      />
      <div
        style={{
          position: "relative",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: "24px",
          width: "320px",
          maxWidth: "90vw",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
          animation: "bzFadeUp 0.15s ease-out",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <Trash2 size={16} style={{ color: "#ef4444", flexShrink: 0 }} />
          <span
            className="bz-oswald"
            style={{ fontSize: "0.9rem", color: "var(--text-primary)" }}
          >
            Delete Climb
          </span>
        </div>
        <p
          className="bz-mono"
          style={{ fontSize: "0.65rem", color: "var(--text-muted)", margin: 0 }}
        >
          Delete{" "}
          <span style={{ color: "var(--text-primary)" }}>
            {climb.name || "Unnamed"}
          </span>
          ? This cannot be undone.
        </p>
        <div
          style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}
        >
          <button
            onClick={onCancel}
            className="bz-mono"
            style={{
              padding: "6px 16px",
              fontSize: "0.6rem",
              background: "transparent",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              color: "var(--text-muted)",
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="bz-mono"
            style={{
              padding: "6px 16px",
              fontSize: "0.6rem",
              background: "#ef4444",
              border: "1px solid #ef4444",
              borderRadius: "var(--radius)",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── ClimbList ───────────────────────────────────────────────────────────────

interface ClimbListProps {
  climbs: Climb[];
  loading: boolean;
  selectedClimb: Climb | null;
  onSelectClimb: (climb: Climb) => void;
  total: number;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  filters: ClimbFilters;
  onFiltersChange: (f: ClimbFilters) => void;
  onFiltersReset: () => void;
  currentUserId: string | null;
  onDeleteClimb: (climb: Climb) => void;
  wallOwnerId: string;
}

function ClimbList({
  climbs,
  loading,
  selectedClimb,
  onSelectClimb,
  total,
  searchQuery,
  onSearchChange,
  filters,
  onFiltersChange,
  onFiltersReset,
  currentUserId,
  onDeleteClimb,
  wallOwnerId,
}: ClimbListProps) {
  const [showFilters, setShowFilters] = useState(false);

  const activeFilterCount = [
    filters.minGrade !== DEFAULT_CLIMB_FILTERS.minGrade,
    filters.maxGrade !== DEFAULT_CLIMB_FILTERS.maxGrade,
    !filters.includeProjects,
    filters.setterName !== "",
    filters.after !== "",
    filters.sortBy !== DEFAULT_CLIMB_FILTERS.sortBy,
    filters.descending !== DEFAULT_CLIMB_FILTERS.descending,
  ].filter(Boolean).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Search + filter toggle bar */}
      <div
        style={{
          padding: "10px 12px",
          borderBottom: showFilters ? "none" : "1px solid var(--border)",
          display: "flex",
          gap: "6px",
          alignItems: "center",
        }}
      >
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            gap: "8px",
            background: "var(--bg)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "6px 10px",
          }}
        >
          <Search
            size={12}
            style={{ color: "var(--text-dim)", flexShrink: 0 }}
          />
          <input
            type="text"
            placeholder="Search climbs…"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--text-primary)",
              fontFamily: "'Space Mono', monospace",
              fontSize: "0.65rem",
            }}
          />
          {searchQuery && (
            <button
              onClick={() => onSearchChange("")}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-dim)",
                cursor: "pointer",
                padding: 0,
                display: "flex",
              }}
            >
              <X size={12} />
            </button>
          )}
        </div>
        {/* Filter toggle button */}
        <button
          onClick={() => setShowFilters((v) => !v)}
          title="Filter & Sort"
          style={{
            position: "relative",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: "30px",
            height: "30px",
            flexShrink: 0,
            border: `1px solid ${showFilters || activeFilterCount > 0 ? "var(--cyan)" : "var(--border)"}`,
            background:
              showFilters || activeFilterCount > 0
                ? "var(--cyan-dim)"
                : "transparent",
            borderRadius: "var(--radius)",
            color:
              showFilters || activeFilterCount > 0
                ? "var(--cyan)"
                : "var(--text-muted)",
            cursor: "pointer",
            transition: "all 0.12s",
          }}
        >
          <SlidersHorizontal size={12} />
          {activeFilterCount > 0 && (
            <div
              style={{
                position: "absolute",
                top: "-4px",
                right: "-4px",
                width: "12px",
                height: "12px",
                borderRadius: "50%",
                background: "var(--cyan)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <span
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: "0.45rem",
                  color: "#09090b",
                  fontWeight: 700,
                }}
              >
                {activeFilterCount}
              </span>
            </div>
          )}
        </button>
      </div>

      {/* Filter panel */}
      {showFilters && (
        <FilterPanel
          filters={filters}
          onChange={onFiltersChange}
          onReset={onFiltersReset}
        />
      )}

      {/* Header count */}
      <div
        style={{
          padding: "8px 12px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <SectionLabel>
          {total} Climb{total !== 1 ? "s" : ""}
        </SectionLabel>
        <span
          className="bz-mono"
          style={{ fontSize: "0.5rem", color: "var(--text-dim)" }}
        >
          {SORT_OPTIONS.find((o) => o.value === filters.sortBy)?.label ??
            "Date"}{" "}
          {filters.descending ? "↓" : "↑"}
        </span>
      </div>

      {/* List body */}
      <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
        {loading ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
            }}
          >
            <Loader2
              size={20}
              style={{
                animation: "spin 1s linear infinite",
                color: "var(--text-dim)",
              }}
            />
          </div>
        ) : climbs.length === 0 ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              padding: "40px 16px",
              gap: "10px",
            }}
          >
            <ChartNetwork size={28} style={{ color: "var(--text-dim)" }} />
            <p
              className="bz-mono"
              style={{
                fontSize: "0.65rem",
                color: "var(--text-muted)",
                textAlign: "center",
              }}
            >
              {searchQuery
                ? "No climbs match your search."
                : "No climbs on this wall yet."}
            </p>
          </div>
        ) : (
          climbs.map((climb) => {
            const isSelected = selectedClimb?.id === climb.id;
            const color = gradeToColor(climb.grade);
            const canDelete =
              currentUserId !== null &&
              (climb.setter_id === currentUserId ||
                wallOwnerId === currentUserId);
            return (
              <div
                key={climb.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  borderBottom: "1px solid var(--border)",
                  borderLeft: `2px solid ${isSelected ? "var(--cyan)" : "transparent"}`,
                  background: isSelected ? "var(--cyan-dim)" : "transparent",
                  transition: "all 0.15s",
                }}
              >
                <button
                  onClick={() => onSelectClimb(climb)}
                  style={{
                    flex: 1,
                    minWidth: 0,
                    textAlign: "left",
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "10px 12px",
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  {/* Grade badge */}
                  <div
                    style={{
                      width: "36px",
                      height: "36px",
                      flexShrink: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      borderRadius: "var(--radius)",
                      background: `${color}15`,
                      border: `1px solid ${color}30`,
                    }}
                  >
                    <span
                      className="bz-mono"
                      style={{ fontSize: "0.6rem", fontWeight: 700, color }}
                    >
                      {gradeToString(climb.grade)}
                    </span>
                  </div>

                  {/* Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      className="bz-oswald"
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--text-primary)",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {climb.name || "Unnamed"}
                    </div>
                    <div
                      className="bz-mono"
                      style={{
                        fontSize: "0.55rem",
                        color: "var(--text-muted)",
                        marginTop: "2px",
                        display: "flex",
                        alignItems: "center",
                        gap: "6px",
                      }}
                    >
                      <span>{climb.ascents} ascents</span>
                      {climb.setter_name && (
                        <>
                          <span style={{ color: "var(--text-dim)" }}>·</span>
                          <span>{climb.setter_name}</span>
                        </>
                      )}
                    </div>
                  </div>
                </button>

                {/* Delete button — only for the climb's setter or the wall owner*/}
                {canDelete && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteClimb(climb);
                    }}
                    title="Delete climb"
                    style={{
                      flexShrink: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: "32px",
                      height: "32px",
                      marginRight: "8px",
                      background: "transparent",
                      border: "1px solid transparent",
                      borderRadius: "var(--radius)",
                      color: "var(--text-dim)",
                      cursor: "pointer",
                      transition: "all 0.12s",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = "#ef4444";
                      e.currentTarget.style.borderColor = "rgba(239,68,68,0.3)";
                      e.currentTarget.style.background = "rgba(239,68,68,0.08)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = "var(--text-dim)";
                      e.currentTarget.style.borderColor = "transparent";
                      e.currentTarget.style.background = "transparent";
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ─── ClimbDetails ────────────────────────────────────────────────────────────

function ClimbDetails({
  climb,
}: {
  climb: Climb;
  displaySettings: DisplaySettings;
}) {
  const createdDate = new Date(climb.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  const color = gradeToColor(climb.grade);

  const detailRow: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "8px 10px",
    background: "var(--bg)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
  };

  return (
    <div
      style={{
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        overflowY: "auto",
      }}
    >
      {/* Name + grade header */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div
          style={{
            width: "48px",
            height: "48px",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "var(--radius)",
            background: `${color}15`,
            border: `1px solid ${color}40`,
          }}
        >
          <span className="bz-oswald" style={{ fontSize: "1rem", color }}>
            {gradeToString(climb.grade)}
          </span>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            className="bz-oswald"
            style={{
              fontSize: "1rem",
              color: "var(--text-primary)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {climb.name || "Unnamed Climb"}
          </div>
          <span
            className="bz-mono"
            style={{ fontSize: "0.6rem", color: "var(--text-muted)" }}
          >
            {climb.ascents} ascents
          </span>
        </div>
      </div>

      {/* Details */}
      {climb.setter_name && (
        <div style={detailRow}>
          <User size={12} style={{ color: "var(--text-dim)", flexShrink: 0 }} />
          <span
            className="bz-mono"
            style={{ fontSize: "0.6rem", color: "var(--text-muted)", flex: 1 }}
          >
            Setter
          </span>
          <span
            className="bz-mono"
            style={{ fontSize: "0.6rem", color: "var(--text-primary)" }}
          >
            {climb.setter_name}
          </span>
        </div>
      )}

      <div style={detailRow}>
        <Calendar
          size={12}
          style={{ color: "var(--text-dim)", flexShrink: 0 }}
        />
        <span
          className="bz-mono"
          style={{ fontSize: "0.6rem", color: "var(--text-muted)", flex: 1 }}
        >
          Created
        </span>
        <span
          className="bz-mono"
          style={{ fontSize: "0.6rem", color: "var(--text-primary)" }}
        >
          {createdDate}
        </span>
      </div>

      {/* Tags */}
      {climb.tags && climb.tags.length > 0 && (
        <div style={detailRow}>
          <Tag size={12} style={{ color: "var(--text-dim)", flexShrink: 0 }} />
          <div
            style={{ display: "flex", flexWrap: "wrap", gap: "4px", flex: 1 }}
          >
            {climb.tags.map((tag) => (
              <span
                key={tag}
                className="bz-mono"
                style={{
                  fontSize: "0.55rem",
                  padding: "2px 8px",
                  background: "var(--surface2)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  color: "var(--text-muted)",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── ViewPage ────────────────────────────────────────────────────────────────

function ViewPage() {
  const navigate = useNavigate();
  const { layoutId: layoutIdParam } = Route.useParams();
  const {
    layout,
    loading: wallLoading,
    waking,
    error: wallError,
  } = useLayout(layoutIdParam);
  if (waking) return <WakingScreen />;
  if (wallLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          background: "#000000",
          color: "#ffffff",
          fontFamily: "'Space Mono', monospace",
        }}
      >
        <Loader2 size={24} style={{ animation: "spin 1s linear infinite" }} />
      </div>
    );
  }
  if (wallError || !layout) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          gap: "16px",
          background: "var(--bg)",
          color: "var(--text-muted)",
          fontFamily: "'Space Mono', monospace",
        }}
      >
        <p>{wallError ?? "Wall not found"}</p>
        <button
          onClick={() => navigate({ to: "/" })}
          style={{
            padding: "8px 16px",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--text-primary)",
            cursor: "pointer",
            borderRadius: "var(--radius)",
            fontFamily: "'Space Mono', monospace",
          }}
        >
          Back to Home
        </button>
      </div>
    );
  }
  return <MainViewPage layout={layout} navigate={navigate} />;
}

// ─── MainViewPage ────────────────────────────────────────────────────────────

function MainViewPage({
  layout,
  navigate,
}: {
  layout: LayoutDetail;
  navigate: ReturnType<typeof useNavigate>;
}) {
  const wallId = layout.metadata.id;
  const wallDimensions = {
    width: layout.metadata.dimensions[0],
    height: layout.metadata.dimensions[1],
  };

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>(
    DEFAULT_DISPLAY_SETTINGS,
  );
  const [showDisplaySettings, setShowDisplaySettings] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilters, setActiveFilters] = useState<ClimbFilters>(
    DEFAULT_CLIMB_FILTERS,
  );
  const [mobilePanel, setMobilePanel] = useState<"none" | "left" | "right">(
    "none",
  );
  const closeMobilePanel = useCallback(() => setMobilePanel("none"), []);

  const { userId } = useBetaZeroAuth();

  const {
    climbs,
    loading,
    total,
    selectedClimb,
    setSelectedClimb,
    removeClimb,
  } = useClimbs(wallId);

  const [climbToDelete, setClimbToDelete] = useState<Climb | null>(null);

  const handleDeleteConfirm = useCallback(async () => {
    if (!climbToDelete) return;
    await removeClimb(climbToDelete.id);
    setClimbToDelete(null);
  }, [climbToDelete, removeClimb]);

  const handleFiltersReset = useCallback(
    () => setActiveFilters(DEFAULT_CLIMB_FILTERS),
    [],
  );

  const selectedHoldset = selectedClimb?.holdset ?? null;

  const handleImageLoad = useCallback(
    (d: { width: number; height: number }) => setImageDimensions(d),
    [],
  );

  const handleSwipeNext = useCallback(() => {
    const currentIdx = selectedClimb
      ? climbs.findIndex((c) => c.id === selectedClimb.id)
      : -1;
    const next = currentIdx < climbs.length - 1 ? currentIdx + 1 : 0;
    setSelectedClimb(climbs[next]);
  }, [climbs, selectedClimb]);

  const handleSwipePrev = useCallback(() => {
    const currentIdx = selectedClimb
      ? climbs.findIndex((c) => c.id === selectedClimb.id)
      : -1;
    const prev = currentIdx > 0 ? currentIdx - 1 : climbs.length - 1;
    setSelectedClimb(climbs[prev]);
  }, [climbs, selectedClimb]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (!climbs.length) return;

      if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        handleSwipeNext();
      } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        handleSwipePrev();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [climbs, selectedClimb, setSelectedClimb]);

  return (
    <>
      <style>{GLOBAL_STYLES}</style>

      {climbToDelete && (
        <DeleteConfirmModal
          climb={climbToDelete}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setClimbToDelete(null)}
        />
      )}

      <div
        style={{
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          background: "var(--bg)",
          color: "var(--text-primary)",
          position: "relative",
        }}
      >
        {/* Header */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 20px",
            height: "48px",
            flexShrink: 0,
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
            zIndex: 20,
          }}
        >
          {/* Left */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <button
              onClick={() => navigate({ to: "/" })}
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
              onMouseEnter={(e) =>
                (e.currentTarget.style.color = "var(--cyan)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.color = "var(--text-muted)")
              }
            >
              <ArrowLeft size={12} />
              <span className="hidden sm:inline">Back</span>
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
              <span
                className="bz-oswald"
                style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}
              >
                {layout.metadata.name}
              </span>
            </div>
          </div>
          {/* Center */}
          <button
            className="bz-mono"
            style={{
              fontSize: "0.55rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--text-primary)",
            }}
            onClick={() =>
              navigate({
                to: "/$layoutId/set",
                params: { layoutId: wallId },
              })
            }
          >
            Set
          </button>
          <a
            href="https://docs.google.com/forms/d/e/1FAIpQLSeYDIel5MMjj0X3zlXFe4N4FZdUcXadAL5bR-Wjb4W7SVZ5SQ/viewform?usp=dialog"
            target="_blank"
            rel="noopener noreferrer"
            className="bz-mono"
            style={{
              fontSize: "0.55rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--text-dim)",
            }}
          >
            Give Feedback
          </a>

          {/* Right: display settings */}
          <div
            style={{
              position: "relative",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <span
              className="bz-mono"
              style={{
                fontSize: "0.55rem",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--text-dim)",
              }}
            >
              Hold Display
            </span>
            <button
              onClick={() => setShowDisplaySettings(!showDisplaySettings)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: "28px",
                height: "28px",
                border: `1px solid ${showDisplaySettings ? "var(--cyan)" : "var(--border)"}`,
                background: showDisplaySettings
                  ? "var(--cyan-dim)"
                  : "transparent",
                color: showDisplaySettings
                  ? "var(--cyan)"
                  : "var(--text-muted)",
                cursor: "pointer",
                borderRadius: "var(--radius)",
                transition: "all 0.15s",
              }}
            >
              <SunMedium size={13} />
            </button>

            {showDisplaySettings && (
              <>
                <div
                  style={{ position: "fixed", inset: 0, zIndex: 40 }}
                  onClick={() => setShowDisplaySettings(false)}
                />
                <div
                  style={{
                    position: "absolute",
                    right: 0,
                    top: "calc(100% + 8px)",
                    padding: "16px",
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
                    zIndex: 50,
                    width: "280px",
                    animation: "bzFadeUp 0.15s ease-out",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginBottom: "16px",
                    }}
                  >
                    <div
                      style={{
                        width: "2px",
                        height: "10px",
                        background: "var(--cyan)",
                      }}
                    />
                    <SectionLabel>Hold Display Settings</SectionLabel>
                  </div>
                  <DisplaySettingsPanel
                    settings={displaySettings}
                    onChange={setDisplaySettings}
                  />
                </div>
              </>
            )}
          </div>
        </header>

        {/* Mobile climb chip */}
        {selectedClimb && (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              position: "absolute",
              top: "56px",
              left: 0,
              right: 0,
              zIndex: 10,
              pointerEvents: "none",
            }}
            className="lg:hidden"
          >
            <div
              style={{
                pointerEvents: "auto",
                background: "rgba(17,17,19,0.92)",
                backdropFilter: "blur(8px)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "8px 18px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                animation: "bzFadeUp 0.2s ease-out",
              }}
            >
              <span
                className="bz-oswald"
                style={{
                  fontSize: "1.6rem",
                  color: gradeToColor(selectedClimb.grade),
                }}
              >
                {selectedClimb.name || "Unnamed"}
              </span>
              <span
                className="bz-mono"
                style={{ fontSize: "0.6rem", color: "var(--cyan)" }}
              >
                {gradeToString(selectedClimb.grade)} @ {selectedClimb.angle} ·{" "}
                {selectedClimb.setter_name}
              </span>
            </div>
          </div>
        )}

        {/* Body */}
        <div
          style={{
            flex: 1,
            display: "flex",
            minHeight: 0,
            position: "relative",
          }}
        >
          {/* Left panel (desktop) — Climb details + list */}
          <div
            style={{
              width: "300px",
              flexShrink: 0,
              flexDirection: "column",
              borderRight: "1px solid var(--border)",
              background: "var(--surface)",
            }}
            className="hidden lg:flex"
          >
            {/* Details section */}
            {selectedClimb && (
              <div
                style={{
                  maxHeight: "340px",
                  borderBottom: "1px solid var(--border)",
                  flexShrink: 0,
                  overflowY: "auto",
                }}
              >
                <ClimbDetails
                  climb={selectedClimb}
                  displaySettings={displaySettings}
                />
              </div>
            )}
            {/* Climb list */}
            <div style={{ flex: 1, minHeight: 0 }}>
              <ClimbList
                climbs={climbs}
                loading={loading}
                selectedClimb={selectedClimb}
                onSelectClimb={setSelectedClimb}
                total={total}
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                filters={activeFilters}
                onFiltersChange={setActiveFilters}
                onFiltersReset={handleFiltersReset}
                currentUserId={userId}
                onDeleteClimb={setClimbToDelete}
                wallOwnerId={layout.metadata.owner_id}
              />
            </div>
          </div>

          {/* Mobile left drawer */}
          {mobilePanel === "left" && (
            <div
              style={{
                position: "fixed",
                inset: 0,
                zIndex: 40,
                display: "flex",
              }}
              className="lg:hidden"
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.7)",
                }}
                onClick={closeMobilePanel}
              />
              <div
                style={{
                  position: "relative",
                  width: "300px",
                  maxWidth: "85vw",
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  background: "var(--surface)",
                  borderRight: "1px solid var(--border)",
                  zIndex: 10,
                  animation: "bzSlideInLeft 0.2s ease-out",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "12px 16px",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                    }}
                  >
                    <div
                      style={{
                        width: "2px",
                        height: "12px",
                        background: "var(--cyan)",
                      }}
                    />
                    <span
                      className="bz-oswald"
                      style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}
                    >
                      Browse Climbs
                    </span>
                  </div>
                  <button
                    onClick={closeMobilePanel}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--text-muted)",
                      cursor: "pointer",
                    }}
                  >
                    <X size={16} />
                  </button>
                </div>
                <div style={{ flex: 1, minHeight: 0 }}>
                  <ClimbList
                    climbs={climbs}
                    loading={loading}
                    selectedClimb={selectedClimb}
                    onSelectClimb={(c) => {
                      setSelectedClimb(c);
                      closeMobilePanel();
                    }}
                    total={total}
                    searchQuery={searchQuery}
                    onSearchChange={setSearchQuery}
                    filters={activeFilters}
                    onFiltersChange={setActiveFilters}
                    onFiltersReset={handleFiltersReset}
                    currentUserId={userId}
                    onDeleteClimb={setClimbToDelete}
                    wallOwnerId={layout.metadata.owner_id}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Canvas */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <WallCanvas
              wallId={wallId}
              holds={layout.holds ?? []}
              wallDimensions={wallDimensions}
              selectedHoldset={selectedHoldset}
              imageDimensions={imageDimensions}
              onImageLoad={handleImageLoad}
              displaySettings={displaySettings}
              imageEdges={
                layout.metadata.image_edges as
                  | [number, number, number, number]
                  | null
              }
            />
          </div>

          {/* Right panel (desktop) — Details when selected */}
          <div
            style={{
              width: "260px",
              flexShrink: 0,
              flexDirection: "column",
              background: "var(--surface)",
              borderLeft: "1px solid var(--border)",
            }}
            className="hidden lg:flex"
          >
            {selectedClimb ? (
              <ClimbDetails
                climb={selectedClimb}
                displaySettings={displaySettings}
              />
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                  padding: "40px 16px",
                  gap: "10px",
                }}
              >
                <ChartNetwork size={28} style={{ color: "var(--text-dim)" }} />
                <p
                  className="bz-mono"
                  style={{
                    fontSize: "0.65rem",
                    color: "var(--text-muted)",
                    textAlign: "center",
                  }}
                >
                  Select a climb to
                  <br />
                  view details.
                </p>
              </div>
            )}
          </div>

          {/* Mobile right drawer (details) */}
          {mobilePanel === "right" && selectedClimb && (
            <div
              style={{
                position: "fixed",
                inset: 0,
                zIndex: 40,
                display: "flex",
                justifyContent: "flex-end",
              }}
              className="lg:hidden"
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(0,0,0,0.7)",
                }}
                onClick={closeMobilePanel}
              />
              <div
                style={{
                  position: "relative",
                  width: "300px",
                  maxWidth: "85vw",
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  background: "var(--surface)",
                  borderLeft: "1px solid var(--border)",
                  zIndex: 10,
                  animation: "bzSlideInRight 0.2s ease-out",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "12px 16px",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                    }}
                  >
                    <div
                      style={{
                        width: "2px",
                        height: "12px",
                        background: "var(--cyan)",
                      }}
                    />
                    <span
                      className="bz-oswald"
                      style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}
                    >
                      Climb Details
                    </span>
                  </div>
                  <button
                    onClick={closeMobilePanel}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--text-muted)",
                      cursor: "pointer",
                    }}
                  >
                    <X size={16} />
                  </button>
                </div>
                <div style={{ flex: 1, overflowY: "auto" }}>
                  <ClimbDetails
                    climb={selectedClimb}
                    displaySettings={displaySettings}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Mobile FABs - climb navigation */}
          <MobileSwipeNav
            count={climbs.length}
            onPrev={handleSwipePrev}
            onNext={handleSwipeNext}
          />
          {/* Mobile FABs */}
          <div
            style={{
              position: "absolute",
              bottom: "48px",
              left: 0,
              right: 0,
              justifyContent: "center",
              gap: "10px",
              zIndex: 30,
              pointerEvents: "none",
              padding: "0 16px",
            }}
            className="flex lg:hidden"
          >
            <button
              onClick={() =>
                setMobilePanel((p) => (p === "left" ? "none" : "left"))
              }
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
                background: "var(--surface)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                cursor: "pointer",
                borderRadius: "var(--radius)",
                boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
              }}
            >
              <ChartNetwork size={12} style={{ color: "var(--cyan)" }} /> Climbs
            </button>

            {false && (
              <button
                onClick={() =>
                  setMobilePanel((p) => (p === "right" ? "none" : "right"))
                }
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
                <Hash size={12} /> Details
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
