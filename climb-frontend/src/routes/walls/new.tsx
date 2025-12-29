import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useCallback, useRef } from "react";
import { useImageCrop } from "@/hooks/useImageCrop";
import { ImageCropper } from "@/components";
import { createWall } from "@/api/walls";

export const Route = createFileRoute("/walls/new")({
  component: NewWallPage,
});

type Step = "upload" | "crop" | "details";

function NewWallPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("upload");
  const [croppedBlob, setCroppedBlob] = useState<Blob | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Form state
  const [name, setName] = useState("");
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [angle, setAngle] = useState("");

  // Image cropping hook
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
    [setImage]
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
    [setImage]
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
        setError("Please provide a wall name");
        return;
      }

      setIsSubmitting(true);
      setError(null);

      try {
        // Create file from blob
        const file = new File([croppedBlob], "wall-photo.jpg", {
          type: "image/jpeg",
        });

        // Parse dimensions if provided
        const dimensions: [number, number] = [
          parseInt(width, 10),
          parseInt(height, 10),
        ];

        const response = await createWall({
          name: name.trim(),
          photo: file,
          dimensions,
          ...(parseInt(angle, 10) && { angle: parseInt(angle, 10) }),
        });

        // Navigate to holds page for the new wall
        navigate({ to: `/walls/${response.id}/holds` });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to create wall";
        setError(message);
        setIsSubmitting(false);
      }
    },
    [croppedBlob, name, width, height, angle, navigate]
  );

  return (
    <div className="min-h-[calc(100vh-4rem)] px-6 py-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate({ to: "/" })}
            className="text-zinc-500 hover:text-zinc-300 transition-colors text-sm mb-4 flex items-center gap-1"
          >
            <span>←</span> Back to walls
          </button>
          <h1 className="text-2xl font-medium">Create New Wall</h1>
          <p className="text-zinc-500 mt-1">
            {step === "upload" && "Upload a photo of your climbing wall"}
            {step === "crop" && "Crop the image to align with the wall edges"}
            {step === "details" && "Add wall details and submit"}
          </p>
        </div>

        {/* Progress indicator */}
        <div className="flex items-center gap-2 mb-8">
          {["upload", "crop", "details"].map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                  step === s
                    ? "bg-purple-600 text-white"
                    : ["upload", "crop", "details"].indexOf(step) > i
                      ? "bg-purple-900 text-purple-300"
                      : "bg-zinc-800 text-zinc-500"
                }`}
              >
                {i + 1}
              </div>
              {i < 2 && (
                <div
                  className={`w-12 h-0.5 ${
                    ["upload", "crop", "details"].indexOf(step) > i
                      ? "bg-purple-600"
                      : "bg-zinc-800"
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Error display */}
        {error && (
          <div className="mb-6 px-4 py-3 bg-red-900/30 border border-red-800 rounded text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Step 1: Upload */}
        {step === "upload" && (
          <div
            className="border-2 border-dashed border-zinc-700 rounded-lg p-12 text-center hover:border-zinc-500 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleFileSelect}
              className="hidden"
            />
            <div className="text-zinc-400 mb-4">
              <svg
                className="w-12 h-12 mx-auto mb-4 text-zinc-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p className="text-lg mb-2">Drop your wall photo here</p>
              <p className="text-sm text-zinc-600">or click to browse</p>
            </div>
            <p className="text-xs text-zinc-600">JPEG or PNG, up to 10MB</p>
          </div>
        )}

        {/* Step 2: Crop */}
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

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleBackToUpload}
                className="px-4 py-2 border border-zinc-700 rounded text-zinc-300 hover:bg-zinc-800 transition-colors"
              >
                Choose Different Photo
              </button>
              <button
                onClick={handleCropConfirm}
                className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white font-medium transition-colors"
              >
                Confirm Crop
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Details */}
        {step === "details" && croppedBlob && (
          <form onSubmit={handleSubmit}>
            {/* Preview */}
            <div className="mb-6">
              <label className="block text-sm text-zinc-400 mb-2">
                Preview
              </label>
              <div className="relative rounded-lg overflow-hidden bg-zinc-900">
                <img
                  src={URL.createObjectURL(croppedBlob)}
                  alt="Cropped wall preview"
                  className="w-full h-auto max-h-64 object-contain"
                />
                <button
                  type="button"
                  onClick={handleBackToCrop}
                  className="absolute top-2 right-2 px-2 py-1 bg-black/70 rounded text-xs text-zinc-300 hover:bg-black transition-colors"
                >
                  Re-crop
                </button>
              </div>
            </div>

            {/* Name field */}
            <div className="mb-4">
              <label
                htmlFor="name"
                className="block text-sm text-zinc-400 mb-2"
              >
                Wall Name <span className="text-red-400">*</span>
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Home Wall"
                required
                className="w-full px-4 py-3 bg-zinc-900 border border-zinc-700 rounded focus:border-purple-500 focus:outline-none text-zinc-100 placeholder-zinc-600"
              />
            </div>

            {/* Dimensions fields */}
            <div className="mb-4">
              <label className="block text-sm text-zinc-400 mb-2">
                Dimensions
              </label>
              <div className="flex gap-3">
                <div className="flex-1">
                  <input
                    type="number"
                    value={width}
                    onChange={(e) => setWidth(e.target.value)}
                    placeholder="Width (ft)"
                    min="1"
                    className="w-full px-4 py-3 bg-zinc-900 border border-zinc-700 rounded focus:border-purple-500 focus:outline-none text-zinc-100 placeholder-zinc-600"
                  />
                </div>
                <span className="flex items-center text-zinc-600">×</span>
                <div className="flex-1">
                  <input
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                    placeholder="Height (ft)"
                    min="1"
                    className="w-full px-4 py-3 bg-zinc-900 border border-zinc-700 rounded focus:border-purple-500 focus:outline-none text-zinc-100 placeholder-zinc-600"
                  />
                </div>
              </div>
            </div>

            {/* Angle field */}
            <div className="mb-8">
              <label
                htmlFor="angle"
                className="block text-sm text-zinc-400 mb-2"
              >
                Angle from vertical (optional)
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="angle"
                  type="number"
                  value={angle}
                  onChange={(e) => setAngle(e.target.value)}
                  placeholder="0"
                  min="-90"
                  max="90"
                  className="w-32 px-4 py-3 bg-zinc-900 border border-zinc-700 rounded focus:border-purple-500 focus:outline-none text-zinc-100 placeholder-zinc-600"
                />
                <span className="text-zinc-500">degrees</span>
              </div>
              <p className="text-xs text-zinc-600 mt-1">
                Positive = overhanging, Negative = slab
              </p>
            </div>

            {/* Submit button */}
            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleBackToCrop}
                className="px-4 py-3 border border-zinc-700 rounded text-zinc-300 hover:bg-zinc-800 transition-colors"
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
                  parseInt(width, 10) <= 0 ||
                  parseInt(height, 10) <= 0
                }
                className="flex-1 px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-900 disabled:cursor-not-allowed rounded text-white font-medium transition-colors"
              >
                {isSubmitting ? "Creating Wall..." : "Create Wall & Add Holds"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
