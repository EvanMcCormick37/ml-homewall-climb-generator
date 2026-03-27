import { fetchLayoutPhoto } from "@/api/layouts";
import type { HoldDetail } from "@/types";
import {
  HOLD_STROKE_COLOR,
  type DisplaySettings,
  type NamedHoldset,
} from "./types";

// ─── URL encoding / decoding ─────────────────────────────────────────────────

export function encodeClimbToParam(entry: NamedHoldset): string {
  const compact = {
    n: entry.name,
    g: entry.grade,
    a: entry.angle,
    s: entry.holdset.start,
    f: entry.holdset.finish,
    h: entry.holdset.hand,
    t: entry.holdset.foot,
  };
  return btoa(JSON.stringify(compact))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

export function decodeClimbFromParam(param: string): NamedHoldset | null {
  try {
    let b64 = param.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4 !== 0) b64 += "=";
    const compact = JSON.parse(atob(b64));
    if (!compact) return null;
    return {
      name: typeof compact.n === "string" ? compact.n : "Unnamed",
      grade: typeof compact.g === "string" ? compact.g : "V?",
      angle: typeof compact.a === "string" ? compact.a : "?",
      holdset: {
        start: Array.isArray(compact.s) ? compact.s : [],
        finish: Array.isArray(compact.f) ? compact.f : [],
        hand: Array.isArray(compact.h) ? compact.h : [],
        foot: Array.isArray(compact.t) ? compact.t : [],
      },
    };
  } catch {
    return null;
  }
}

export function buildShareUrl(layoutId: string, entry: NamedHoldset): string {
  return `${window.location.origin}/${layoutId}/set?climb=${encodeClimbToParam(entry)}`;
}

// ─── Export image renderer ───────────────────────────────────────────────────

export async function renderExportImage(
  layoutId: string,
  layoutName: string,
  holds: HoldDetail[],
  layoutDimensions: { width: number; height: number },
  climb: NamedHoldset,
  setterName: string | null,
  displaySettings: DisplaySettings,
  imageEdges?: [number, number, number, number] | null,
): Promise<Blob> {
  const objectUrl = await fetchLayoutPhoto(layoutId);
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new window.Image();
    el.onload = () => resolve(el);
    el.onerror = reject;
    el.src = objectUrl;
  });
  URL.revokeObjectURL(objectUrl);

  const imgW = img.width,
    imgH = img.height;
  const topBannerH = Math.round(imgH * 0.06);
  const bottomBannerH = Math.round(imgH * 0.045);
  const canvas = document.createElement("canvas");
  canvas.width = imgW;
  canvas.height = imgH + topBannerH + bottomBannerH;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111113";
  ctx.fillRect(0, 0, imgW, topBannerH);

  // 1. Draw the Climb Name
  ctx.fillStyle = "#f4f4f5";
  ctx.font = `bold ${Math.round(topBannerH * 0.55)}px sans-serif`;
  ctx.textBaseline = "middle";
  const nameStartX = Math.round(imgW * 0.02);
  ctx.fillText(climb.name, nameStartX, topBannerH / 2);

  // 2. Format and Draw the Subtitle Text
  // Measure the width of the name we just drew to know where to start the subtitle
  const nameWidth = ctx.measureText(climb.name).width;
  const subtitleGap = Math.round(imgW * 0.015); // Gap between name and subtitle

  // Handle nulls and formatting
  const gradeStr = climb.grade !== null ? climb.grade : "?";
  const setterStr = setterName || "Unknown";
  const dateStr = new Date().toLocaleString();

  const subtitle = `${gradeStr} @ ${climb.angle}° | ${setterStr}, ${dateStr}`;

  ctx.fillStyle = "#a1a1aa"; // A muted gray color so it doesn't fight the main title
  ctx.font = `normal ${Math.round(topBannerH * 0.4)}px sans-serif`;
  ctx.fillText(subtitle, nameStartX + nameWidth + subtitleGap, topBannerH / 2);

  // 3. Draw the Wall Name (Right aligned)
  ctx.fillStyle = "#71717a";
  ctx.font = `${Math.round(topBannerH * 0.4)}px sans-serif`;
  ctx.textAlign = "right";
  ctx.fillText(layoutName, imgW - Math.round(imgW * 0.02), topBannerH / 2);

  // Reset alignment before drawing the image
  ctx.textAlign = "left";
  ctx.drawImage(img, 0, topBannerH);

  const startSet = new Set(climb.holdset.start),
    finishSet = new Set(climb.holdset.finish);
  const handSet = new Set(climb.holdset.hand),
    footSet = new Set(climb.holdset.foot);
  const usedSet = new Set([...startSet, ...finishSet, ...handSet, ...footSet]);
  const {
    scale: userScale,
    colorMode,
    uniformColor,
    opacity: userOpacity,
    filled,
  } = displaySettings;

  const [imgEdgeL, imgEdgeR, imgEdgeB, imgEdgeT] = imageEdges ?? [
    0,
    layoutDimensions.width,
    0,
    layoutDimensions.height,
  ];

  holds.forEach((hold) => {
    const px = ((hold.x - imgEdgeL) / (imgEdgeR - imgEdgeL)) * imgW;
    const py = ((imgEdgeT - hold.y) / (imgEdgeT - imgEdgeB)) * imgH + topBannerH;
    const baseScale = imgH / 500;
    const radius = 10 * baseScale * userScale;
    const isUsed = usedSet.has(hold.hold_index);
    const isStart = startSet.has(hold.hold_index),
      isFinish = finishSet.has(hold.hold_index);
    const isHand = handSet.has(hold.hold_index),
      isFoot = footSet.has(hold.hold_index);
    const baseAlpha = isUsed ? 1 : 0.15;
    const alpha = isUsed ? baseAlpha * userOpacity : baseAlpha;
    let color = HOLD_STROKE_COLOR;
    if (isUsed) {
      if (colorMode === "uniform") color = uniformColor;
      else if (isStart) color = displaySettings.categoryColors.start;
      else if (isFinish) color = displaySettings.categoryColors.finish;
      else if (isHand) color = displaySettings.categoryColors.hand;
      else if (isFoot) color = displaySettings.categoryColors.foot;
    }
    const footScale = isFoot ? 0.5 : 1;
    ctx.beginPath();
    ctx.arc(px, py, radius * footScale, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = isUsed ? baseScale * 2 : 2;
    if (isUsed && filled) {
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  });

  const legendY = topBannerH + imgH;
  ctx.fillStyle = "#111113";
  ctx.fillRect(0, legendY, imgW, bottomBannerH);
  const legendFont = Math.round(bottomBannerH * 0.45);
  ctx.font = `${legendFont}px sans-serif`;
  ctx.textBaseline = "middle";
  const legendMidY = legendY + bottomBannerH / 2;
  const dotR = Math.round(bottomBannerH * 0.15);
  const pad = Math.round(imgW * 0.02);
  const legendItems = [
    { label: "Start", color: displaySettings.categoryColors.start },
    { label: "Finish", color: displaySettings.categoryColors.finish },
    { label: "Hand", color: displaySettings.categoryColors.hand },
    { label: "Foot", color: displaySettings.categoryColors.foot },
  ];
  let cursorX = pad;
  for (const item of legendItems) {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(cursorX + dotR, legendMidY, dotR, 0, 2 * Math.PI);
    ctx.fill();
    cursorX += dotR * 2 + 6;
    ctx.fillStyle = "#a1a1aa";
    ctx.fillText(item.label, cursorX, legendMidY);
    cursorX += ctx.measureText(item.label).width + pad;
  }

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("toBlob failed"))),
      "image/png",
    );
  });
}
