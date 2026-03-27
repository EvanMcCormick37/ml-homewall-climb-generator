import { memo, useEffect, useRef } from "react";

const SPACING = 48;
type Dot = { cx: number; cy: number; r: number; a: number };

const HoldGridCanvas = memo(function HoldGridCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let dots: Dot[] = [];

    const generateDots = (w: number, h: number) => {
      const cols = Math.ceil(w / SPACING) + 1;
      const rows = Math.ceil(h / SPACING) + 1;
      dots = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const jitter = SPACING * 0.18;
          dots.push({
            cx: c * SPACING + (Math.random() - 0.5) * jitter,
            cy: r * SPACING + (Math.random() - 0.5) * jitter,
            r: 2 + Math.random() * 3,
            a: 0.04 + Math.random() * 0.09,
          });
        }
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      dots.forEach(({ cx, cy, r, a }) => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6,182,212,${a})`;
        ctx.fill();
      });
    };

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    generateDots(canvas.width, canvas.height);
    draw();

    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    const handleResize = () => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        draw();
        resizeTimer = null;
      }, 150);
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      if (resizeTimer) clearTimeout(resizeTimer);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    />
  );
});

export default HoldGridCanvas;
