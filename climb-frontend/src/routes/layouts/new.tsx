import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef } from "react";
import { useUser } from "@clerk/clerk-react";
import { useImageCrop } from "@/hooks/useImageCrop";
import { ImageCropper } from "@/components";
import { createLayout } from "@/api/layouts";
import { createSize } from "@/api/sizes";
import { ArrowLeft, Globe, Lock, Link } from "lucide-react";
import type { Visibility } from "@/types";

export const Route = createFileRoute("/layouts/new")({
  component: NewLayoutPage,
});

const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&family=Space+Mono:wght@400;700&display=swap');

  :root {
    --cyan: #06b6d4;
    --cyan-dim: rgba(6,182,212,0.15);
    --bg: #09090b;
    --surface: #111113;
    --surface2: #1c1c1e;
    --border: rgba(255,255,255,0.08);
    --border-active: rgba(6,182,212,0.4);
    --text-primary: #f4f4f5;
    --text-muted: #71717a;
    --text-dim: #52525b;
    --radius: 4px;
  }

  .bz-oswald { font-family: 'Oswald', sans-serif; }
  .bz-mono { font-family: 'Space Mono', monospace; }
`;

type Step = "visibility" | "upload" | "crop" | "details";

const STEPS: Step[] = ["visibility", "upload", "crop", "details"];
const STEP_LABELS: Record<Step, string> = {
  visibility: "Access",
  upload: "Upload",
  crop: "Crop",
  details: "Details",
};

const VISIBILITY_OPTIONS: {
  value: Visibility;
  label: string;
  icon: React.ReactNode;
  description: string;
  detail: string;
}[] = [
  {
    value: "public",
    label: "Public",
    icon: <Globe size={22} />,
    description: "Open to everyone",
    detail:
      "Anyone can view this layout and set climbs on it. Ideal for community boards.",
  },
  {
    value: "unlisted",
    label: "Unlisted",
    icon: <Link size={22} />,
    description: "Accessible via link",
    detail:
      "Not listed publicly, but any user with the link can view it and set climbs.",
  },
  {
    value: "private",
    label: "Private",
    icon: <Lock size={22} />,
    description: "Only you",
    detail:
      "Accessible only via a private share token. Only you can set climbs on it.",
  },
];

function NewLayoutPage() {
  const navigate = useNavigate();
  const { isSignedIn, isLoaded } = useUser();
  const [step, setStep] = useState<Step>("visibility");
  const [visibility, setVisibility] = useState<Visibility>("public");
  const [croppedBlob, setCroppedBlob] = useState<Blob | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [name, setName] = useState("");
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [angle, setAngle] = useState("");

  const {
    imageUrl,
    cropArea,
    isDragging,
    setImage,
    startDrag,
    updateDrag,
    endDrag,
    resetCrop,
    getCroppedImage,
  } = useImageCrop();

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        if (!file.type.startsWith("image/")) {
          setError("Please select an image file (JPEG or PNG)");
          return;
        }
        setError(null);
        setImage(file);
        setStep("crop");
      }
    },
    [setImage],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) {
        if (!file.type.startsWith("image/")) {
          setError("Please select an image file (JPEG or PNG)");
          return;
        }
        setError(null);
        setImage(file);
        setStep("crop");
      }
    },
    [setImage],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleCropConfirm = useCallback(async () => {
    const blob = await getCroppedImage();
    if (blob) {
      setCroppedBlob(blob);
      setStep("details");
    }
  }, [getCroppedImage]);

  const handleBackToCrop = useCallback(() => {
    setCroppedBlob(null);
    setStep("crop");
  }, []);

  const handleBackToUpload = useCallback(() => {
    resetCrop();
    setCroppedBlob(null);
    setStep("upload");
  }, [resetCrop]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      if (!croppedBlob || !name.trim()) {
        setError("Please provide a layout name");
        return;
      }

      setIsSubmitting(true);
      setError(null);

      try {
        // Step 1: create the layout (no photo yet)
        const layoutResponse = await createLayout({
          name: name.trim(),
          visibility,
        });

        // Step 2: create first size with the photo and dimensions
        const photoFile = new File([croppedBlob], "wall-photo.jpg", {
          type: "image/jpeg",
        });
        const widthFt = parseFloat(width);
        const heightFt = parseFloat(height);
        const angleDeg = parseInt(angle, 10);

        await createSize(layoutResponse.id, {
          name: `${widthFt}×${heightFt} ft`,
          width_ft: widthFt || undefined,
          height_ft: heightFt || undefined,
          ...(angleDeg > 0 && { edge_top: angleDeg }), // store angle hint in edge_top for now
          photo: photoFile,
        });

        navigate({
          to: "/$layoutId/holds",
          params: { layoutId: layoutResponse.id },
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to create layout";
        setError(message);
        setIsSubmitting(false);
      }
    },
    [croppedBlob, name, width, height, angle, visibility, navigate],
  );

  const inputStyle: React.CSSProperties = {
    width: "100%",
    background: "var(--surface2)",
    color: "var(--text-primary)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "10px 14px",
    fontFamily: "'Space Mono', monospace",
    fontSize: "0.75rem",
    outline: "none",
    boxSizing: "border-box",
  };

  if (!isLoaded) {
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
          Loading…
        </div>
      </>
    );
  }

  if (!isSignedIn) {
    return (
      <>
        <style>{GLOBAL_STYLES}</style>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "24px",
            background: "var(--bg)",
            color: "var(--text-primary)",
          }}
        >
          <h2
            className="bz-oswald"
            style={{
              fontSize: "1.5rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Sign in to add a layout
          </h2>
          <div style={{ display: "flex", gap: "12px" }}>
            <button
              onClick={() => navigate({ to: "/signIn" })}
              style={{
                padding: "9px 24px",
                background: "var(--cyan)",
                color: "#09090b",
                border: "none",
                borderRadius: "var(--radius)",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.7rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: "pointer",
                fontWeight: 700,
              }}
            >
              Sign In
            </button>
            <button
              onClick={() => navigate({ to: "/" })}
              style={{
                padding: "9px 24px",
                background: "transparent",
                color: "var(--text-muted)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                fontFamily: "'Space Mono', monospace",
                fontSize: "0.7rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              Back
            </button>
          </div>
        </div>
      </>
    );
  }

  const stepIndex = STEPS.indexOf(step);

  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <div
        style={{
          minHeight: "100vh",
          background: "var(--bg)",
          color: "var(--text-primary)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <header
          style={{
            padding: "16px 32px",
            borderBottom: "1px solid var(--border)",
            background: "var(--surface)",
            display: "flex",
            alignItems: "center",
            gap: "16px",
          }}
        >
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
              style={{ width: "2px", height: "14px", background: "var(--cyan)" }}
            />
            <span
              className="bz-oswald"
              style={{ fontSize: "0.85rem", color: "var(--text-primary)" }}
            >
              Add Your Wall
            </span>
          </div>
        </header>

        {/* Body */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            padding: "48px 32px",
          }}
        >
          <div style={{ width: "100%", maxWidth: "640px" }}>
            {/* Step subtitle */}
            <p
              className="bz-mono"
              style={{
                fontSize: "0.65rem",
                color: "var(--text-muted)",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: "32px",
              }}
            >
              {step === "visibility" && "Choose who can access this layout"}
              {step === "upload" && "Upload a photo of your climbing wall"}
              {step === "crop" && "Crop the image to align with the wall edges"}
              {step === "details" && "Add wall details and submit"}
            </p>

            {/* Progress indicator */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                marginBottom: "40px",
              }}
            >
              {STEPS.map((s, i) => (
                <div
                  key={s}
                  style={{ display: "flex", alignItems: "center", gap: "8px" }}
                >
                  <div
                    style={{
                      width: "28px",
                      height: "28px",
                      borderRadius: "50%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "0.65rem",
                      fontWeight: 700,
                      transition: "all 0.2s",
                      background:
                        step === s
                          ? "var(--cyan)"
                          : stepIndex > i
                            ? "var(--cyan-dim)"
                            : "var(--surface2)",
                      color:
                        step === s
                          ? "#09090b"
                          : stepIndex > i
                            ? "var(--cyan)"
                            : "var(--text-muted)",
                      border:
                        step === s
                          ? "1px solid var(--cyan)"
                          : stepIndex > i
                            ? "1px solid var(--border-active)"
                            : "1px solid var(--border)",
                    }}
                  >
                    {i + 1}
                  </div>
                  <span
                    className="bz-mono"
                    style={{
                      fontSize: "0.55rem",
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                      color: step === s ? "var(--cyan)" : "var(--text-muted)",
                    }}
                  >
                    {STEP_LABELS[s]}
                  </span>
                  {i < STEPS.length - 1 && (
                    <div
                      style={{
                        width: "32px",
                        height: "1px",
                        background:
                          stepIndex > i ? "var(--cyan)" : "var(--border)",
                        transition: "background 0.2s",
                      }}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Error */}
            {error && (
              <div
                className="bz-mono"
                style={{
                  marginBottom: "24px",
                  padding: "10px 14px",
                  background: "rgba(248,113,113,0.08)",
                  border: "1px solid rgba(248,113,113,0.2)",
                  borderRadius: "var(--radius)",
                  fontSize: "0.65rem",
                  color: "#f87171",
                }}
              >
                {error}
              </div>
            )}

            {/* ── Step 0: Visibility ── */}
            {step === "visibility" && (
              <div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px",
                    marginBottom: "32px",
                  }}
                >
                  {VISIBILITY_OPTIONS.map((opt) => {
                    const isSelected = visibility === opt.value;
                    return (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setVisibility(opt.value)}
                        style={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: "18px",
                          padding: "20px 24px",
                          background: isSelected
                            ? "var(--cyan-dim)"
                            : "var(--surface)",
                          border: `1px solid ${isSelected ? "var(--cyan)" : "var(--border)"}`,
                          borderRadius: "var(--radius)",
                          cursor: "pointer",
                          textAlign: "left",
                          transition: "all 0.15s",
                          width: "100%",
                        }}
                        onMouseEnter={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.borderColor =
                              "rgba(6,182,212,0.3)";
                            e.currentTarget.style.background = "var(--surface2)";
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.borderColor = "var(--border)";
                            e.currentTarget.style.background = "var(--surface)";
                          }
                        }}
                      >
                        <div
                          style={{
                            flexShrink: 0,
                            marginTop: "2px",
                            color: isSelected ? "var(--cyan)" : "var(--text-dim)",
                            transition: "color 0.15s",
                          }}
                        >
                          {opt.icon}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "10px",
                              marginBottom: "4px",
                            }}
                          >
                            <span
                              className="bz-oswald"
                              style={{
                                fontSize: "1.05rem",
                                fontWeight: 700,
                                textTransform: "uppercase",
                                letterSpacing: "0.06em",
                                color: isSelected
                                  ? "var(--cyan)"
                                  : "var(--text-primary)",
                              }}
                            >
                              {opt.label}
                            </span>
                            <span
                              className="bz-mono"
                              style={{
                                fontSize: "0.6rem",
                                letterSpacing: "0.08em",
                                color: isSelected
                                  ? "var(--cyan)"
                                  : "var(--text-dim)",
                              }}
                            >
                              — {opt.description}
                            </span>
                          </div>
                          <p
                            className="bz-mono"
                            style={{
                              fontSize: "0.65rem",
                              lineHeight: 1.65,
                              color: "var(--text-muted)",
                              margin: 0,
                            }}
                          >
                            {opt.detail}
                          </p>
                        </div>
                        <div
                          style={{
                            flexShrink: 0,
                            width: "16px",
                            height: "16px",
                            borderRadius: "50%",
                            border: `2px solid ${isSelected ? "var(--cyan)" : "var(--border)"}`,
                            background: isSelected ? "var(--cyan)" : "transparent",
                            marginTop: "3px",
                            transition: "all 0.15s",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          {isSelected && (
                            <div
                              style={{
                                width: "6px",
                                height: "6px",
                                borderRadius: "50%",
                                background: "#09090b",
                              }}
                            />
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
                <button
                  onClick={() => setStep("upload")}
                  style={{
                    width: "100%",
                    padding: "11px 18px",
                    background: "var(--cyan)",
                    color: "#09090b",
                    border: "none",
                    borderRadius: "var(--radius)",
                    fontFamily: "'Oswald', sans-serif",
                    fontSize: "0.85rem",
                    letterSpacing: "0.1em",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    cursor: "pointer",
                    transition: "opacity 0.15s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
                  onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
                >
                  Continue
                </button>
              </div>
            )}

            {/* ── Step 1: Upload ── */}
            {step === "upload" && (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                style={{
                  border: "1px dashed var(--border)",
                  borderRadius: "var(--radius)",
                  padding: "64px 32px",
                  textAlign: "center",
                  cursor: "pointer",
                  transition: "border-color 0.2s",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.borderColor = "var(--cyan)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.borderColor = "var(--border)")
                }
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  onChange={handleFileSelect}
                  style={{ display: "none" }}
                />
                <svg
                  width="40"
                  height="40"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  style={{
                    margin: "0 auto 16px",
                    color: "var(--text-dim)",
                    display: "block",
                  }}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                <p
                  className="bz-mono"
                  style={{
                    fontSize: "0.85rem",
                    color: "var(--text-muted)",
                    marginBottom: "8px",
                  }}
                >
                  Drop your wall photo here
                </p>
                <p
                  className="bz-mono"
                  style={{ fontSize: "0.65rem", color: "var(--text-dim)" }}
                >
                  or click to browse · JPEG or PNG
                </p>
              </div>
            )}

            {/* ── Step 2: Crop ── */}
            {step === "crop" && imageUrl && cropArea && (
              <div>
                <ImageCropper
                  imageUrl={imageUrl}
                  cropArea={cropArea}
                  isDragging={isDragging}
                  onStartDrag={startDrag}
                  onUpdateDrag={updateDrag}
                  onEndDrag={endDrag}
                />
                <div
                  style={{ display: "flex", gap: "12px", marginTop: "20px" }}
                >
                  <button
                    onClick={handleBackToUpload}
                    style={{
                      padding: "9px 18px",
                      background: "transparent",
                      color: "var(--text-muted)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius)",
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "0.65rem",
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-active)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--border)";
                      e.currentTarget.style.color = "var(--text-muted)";
                    }}
                  >
                    Choose Different Photo
                  </button>
                  <button
                    onClick={handleCropConfirm}
                    style={{
                      flex: 1,
                      padding: "9px 18px",
                      background: "var(--cyan)",
                      color: "#09090b",
                      border: "none",
                      borderRadius: "var(--radius)",
                      fontFamily: "'Oswald', sans-serif",
                      fontSize: "0.85rem",
                      letterSpacing: "0.1em",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.opacity = "0.85")
                    }
                    onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
                  >
                    Confirm Crop
                  </button>
                </div>
              </div>
            )}

            {/* ── Step 3: Details ── */}
            {step === "details" && croppedBlob && (
              <form onSubmit={handleSubmit}>
                {/* Preview */}
                <div style={{ marginBottom: "24px" }}>
                  <label
                    className="bz-mono"
                    style={{
                      display: "block",
                      fontSize: "0.6rem",
                      letterSpacing: "0.15em",
                      textTransform: "uppercase",
                      color: "var(--text-muted)",
                      marginBottom: "8px",
                    }}
                  >
                    Preview
                  </label>
                  <div
                    style={{
                      position: "relative",
                      borderRadius: "var(--radius)",
                      overflow: "hidden",
                      background: "var(--surface2)",
                      border: "1px solid var(--border)",
                    }}
                  >
                    <img
                      src={URL.createObjectURL(croppedBlob)}
                      alt="Cropped wall preview"
                      style={{
                        width: "100%",
                        maxHeight: "240px",
                        objectFit: "contain",
                        display: "block",
                      }}
                    />
                    <button
                      type="button"
                      onClick={handleBackToCrop}
                      style={{
                        position: "absolute",
                        top: "8px",
                        right: "8px",
                        padding: "4px 10px",
                        background: "rgba(0,0,0,0.75)",
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius)",
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "0.6rem",
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        color: "var(--text-muted)",
                        cursor: "pointer",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.color = "var(--cyan)")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.color = "var(--text-muted)")
                      }
                    >
                      Re-crop
                    </button>
                  </div>
                </div>

                {/* Layout Name */}
                <div style={{ marginBottom: "16px" }}>
                  <label
                    className="bz-mono"
                    htmlFor="layout-name"
                    style={{
                      display: "block",
                      fontSize: "0.6rem",
                      letterSpacing: "0.15em",
                      textTransform: "uppercase",
                      color: "var(--text-muted)",
                      marginBottom: "8px",
                    }}
                  >
                    Wall Name <span style={{ color: "#f87171" }}>*</span>
                  </label>
                  <input
                    id="layout-name"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="My Home Wall"
                    required
                    style={{
                      ...inputStyle,
                      transition: "border-color 0.15s",
                    }}
                    onFocus={(e) =>
                      (e.currentTarget.style.borderColor = "var(--border-active)")
                    }
                    onBlur={(e) =>
                      (e.currentTarget.style.borderColor = "var(--border)")
                    }
                  />
                </div>

                {/* Dimensions */}
                <div style={{ marginBottom: "16px" }}>
                  <label
                    className="bz-mono"
                    style={{
                      display: "block",
                      fontSize: "0.6rem",
                      letterSpacing: "0.15em",
                      textTransform: "uppercase",
                      color: "var(--text-muted)",
                      marginBottom: "8px",
                    }}
                  >
                    Wall Dimensions (exact height and width of the cropped area,
                    in feet.)
                  </label>
                  <div
                    style={{ display: "flex", alignItems: "center", gap: "10px" }}
                  >
                    <input
                      type="number"
                      value={width}
                      onChange={(e) => setWidth(e.target.value)}
                      placeholder="Width (ft)"
                      min="1"
                      style={{
                        ...inputStyle,
                        flex: 1,
                        transition: "border-color 0.15s",
                      }}
                      onFocus={(e) =>
                        (e.currentTarget.style.borderColor =
                          "var(--border-active)")
                      }
                      onBlur={(e) =>
                        (e.currentTarget.style.borderColor = "var(--border)")
                      }
                    />
                    <span
                      style={{
                        color: "var(--text-dim)",
                        fontFamily: "'Space Mono', monospace",
                        fontSize: "0.75rem",
                        flexShrink: 0,
                      }}
                    >
                      ×
                    </span>
                    <input
                      type="number"
                      value={height}
                      onChange={(e) => setHeight(e.target.value)}
                      placeholder="Height (ft)"
                      min="1"
                      style={{
                        ...inputStyle,
                        flex: 1,
                        transition: "border-color 0.15s",
                      }}
                      onFocus={(e) =>
                        (e.currentTarget.style.borderColor =
                          "var(--border-active)")
                      }
                      onBlur={(e) =>
                        (e.currentTarget.style.borderColor = "var(--border)")
                      }
                    />
                  </div>
                </div>

                {/* Angle */}
                <div style={{ marginBottom: "32px" }}>
                  <label
                    className="bz-mono"
                    htmlFor="wall-angle"
                    style={{
                      display: "block",
                      fontSize: "0.6rem",
                      letterSpacing: "0.15em",
                      textTransform: "uppercase",
                      color: "var(--text-muted)",
                      marginBottom: "8px",
                    }}
                  >
                    Angle from vertical{" "}
                    <span style={{ color: "var(--text-dim)" }}>(optional)</span>
                  </label>
                  <div
                    style={{ display: "flex", alignItems: "center", gap: "10px" }}
                  >
                    <input
                      id="wall-angle"
                      type="number"
                      value={angle}
                      onChange={(e) => setAngle(e.target.value)}
                      placeholder="0"
                      min="-90"
                      max="90"
                      style={{
                        ...inputStyle,
                        width: "120px",
                        transition: "border-color 0.15s",
                      }}
                      onFocus={(e) =>
                        (e.currentTarget.style.borderColor =
                          "var(--border-active)")
                      }
                      onBlur={(e) =>
                        (e.currentTarget.style.borderColor = "var(--border)")
                      }
                    />
                    <span
                      className="bz-mono"
                      style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}
                    >
                      degrees
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: "12px" }}>
                  <button
                    type="button"
                    onClick={handleBackToCrop}
                    style={{
                      padding: "10px 18px",
                      background: "transparent",
                      color: "var(--text-muted)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius)",
                      fontFamily: "'Space Mono', monospace",
                      fontSize: "0.65rem",
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-active)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--border)";
                      e.currentTarget.style.color = "var(--text-muted)";
                    }}
                  >
                    Back
                  </button>
                  <button
                    type="submit"
                    disabled={
                      isSubmitting ||
                      !name.trim() ||
                      !width ||
                      !height ||
                      parseFloat(width) <= 0 ||
                      parseFloat(height) <= 0
                    }
                    style={{
                      flex: 1,
                      padding: "10px 18px",
                      background:
                        isSubmitting || !name.trim() || !width || !height
                          ? "var(--surface2)"
                          : "var(--cyan)",
                      color:
                        isSubmitting || !name.trim() || !width || !height
                          ? "var(--text-muted)"
                          : "#09090b",
                      border: "none",
                      borderRadius: "var(--radius)",
                      fontFamily: "'Oswald', sans-serif",
                      fontSize: "0.85rem",
                      letterSpacing: "0.1em",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      cursor:
                        isSubmitting || !name.trim() || !width || !height
                          ? "not-allowed"
                          : "pointer",
                      transition: "all 0.15s",
                    }}
                  >
                    {isSubmitting ? "Creating…" : "Create Wall & Add Holds"}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
