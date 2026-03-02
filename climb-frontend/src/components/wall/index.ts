// Styles
export { GLOBAL_STYLES } from "./styles";

// Types & constants
export {
  CATEGORY_ORDER,
  CATEGORY_COLORS,
  CATEGORY_LABELS,
  HOLD_STROKE_COLOR,
  DEFAULT_DISPLAY_SETTINGS,
  type HoldCategory,
  type ColorMode,
  type DisplaySettings,
  type NamedHoldset,
} from "./types";

export { VGRADE_OPTIONS, FONT_OPTIONS, generateClimbName } from "./constants";

// UI primitives
export { SectionLabel, TogglePair, BzRange } from "./ui";

// Components
export { MobileSwipeNav } from "./MobileSwipeNav";
export { WallCanvas, type WallCanvasProps } from "./WallCanvas";
export { DisplaySettingsPanel } from "./DisplaySettingsPanel";
export {
  SaveShareMenu,
  DesktopSaveSharePanel,
  type SaveShareMenuProps,
} from "./SaveShareMenu";

// Sharing utilities
export {
  encodeClimbToParam,
  decodeClimbFromParam,
  buildShareUrl,
  renderExportImage,
} from "./sharing";
