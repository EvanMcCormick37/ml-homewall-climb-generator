import { ChevronLeft, ChevronRight } from "lucide-react";

const btnStyle: React.CSSProperties = {
  background: "rgba(17,17,19,0.92)",
  backdropFilter: "blur(8px)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  width: "36px",
  height: "36px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "var(--text-primary)",
  cursor: "pointer",
};

export function MobileSwipeNav({
  onPrev,
  onNext,
  count,
}: {
  onPrev: () => void;
  onNext: () => void;
  count: number;
}) {
  if (count <= 1) return null;
  return (
    <div
      style={{
        position: "absolute",
        bottom: "96px",
        left: 0,
        right: 0,
        justifyContent: "center",
        gap: "10px",
        zIndex: 30,
        pointerEvents: "auto",
        padding: "0 16px",
      }}
      className="flex lg:hidden"
    >
      <button onClick={onPrev} style={btnStyle}>
        <ChevronLeft size={18} />
      </button>
      <button onClick={onNext} style={btnStyle}>
        <ChevronRight size={18} />
      </button>
    </div>
  );
}
