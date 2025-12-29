import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { getWall, getWallPhotoUrl, deleteWall } from "@/api/walls";
import {
  ArrowLeft,
  Eye,
  Plus,
  Pencil,
  Trash2,
  Grid3X3,
  Mountain,
  Ruler,
  RotateCcw,
} from "lucide-react";
import type { WallDetail } from "@/types";

export const Route = createFileRoute("/walls/$wallId/")({
  component: WallDetailPage,
  loader: async ({ params }) => {
    const wall = await getWall(params.wallId);
    return { wall };
  },
});

function WallDetailPage() {
  const navigate = useNavigate();
  const { wall } = Route.useLoaderData() as { wall: WallDetail };
  const { metadata, holds } = wall;

  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleDelete = async () => {
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await deleteWall(metadata.id);
      navigate({ to: "/" });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete wall";
      setDeleteError(message);
      setIsDeleting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate({ to: "/" })}
            className="flex items-center gap-1 text-zinc-400 hover:text-zinc-100 transition-colors text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <div className="w-px h-5 bg-zinc-700" />
          <h1 className="text-lg font-medium text-zinc-100">{metadata.name}</h1>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Wall info card */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
            <div className="p-6">
              <div className="flex gap-6">
                {/* Wall photo */}
                <div className="w-56 h-56 bg-zinc-800 rounded-lg overflow-hidden flex-shrink-0 border border-zinc-700">
                  <img
                    src={getWallPhotoUrl(metadata.id)}
                    alt={metadata.name}
                    className="w-full h-full object-cover"
                  />
                </div>

                {/* Wall details */}
                <div className="flex-1">
                  <h2 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-4">
                    Wall Details
                  </h2>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
                      <div className="flex items-center gap-2 text-zinc-500 mb-1">
                        <Grid3X3 className="w-4 h-4" />
                        <span className="text-xs uppercase tracking-wider">
                          Holds
                        </span>
                      </div>
                      <div className="text-2xl font-semibold text-zinc-100">
                        {holds?.length ?? 0}
                      </div>
                    </div>

                    <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
                      <div className="flex items-center gap-2 text-zinc-500 mb-1">
                        <Mountain className="w-4 h-4" />
                        <span className="text-xs uppercase tracking-wider">
                          Climbs
                        </span>
                      </div>
                      <div className="text-2xl font-semibold text-zinc-100">
                        {metadata.num_climbs ?? 0}
                      </div>
                    </div>

                    {metadata.dimensions && (
                      <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
                        <div className="flex items-center gap-2 text-zinc-500 mb-1">
                          <Ruler className="w-4 h-4" />
                          <span className="text-xs uppercase tracking-wider">
                            Dimensions
                          </span>
                        </div>
                        <div className="text-2xl font-semibold text-zinc-100">
                          {metadata.dimensions[0]} × {metadata.dimensions[1]}{" "}
                          <span className="text-sm text-zinc-500">ft</span>
                        </div>
                      </div>
                    )}

                    {metadata.angle !== undefined &&
                      metadata.angle !== null && (
                        <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
                          <div className="flex items-center gap-2 text-zinc-500 mb-1">
                            <RotateCcw className="w-4 h-4" />
                            <span className="text-xs uppercase tracking-wider">
                              Angle
                            </span>
                          </div>
                          <div className="text-2xl font-semibold text-zinc-100">
                            {metadata.angle}°
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
            <h2 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-4">
              Quick Actions
            </h2>
            <div className="grid grid-cols-3 gap-4">
              <Link
                to="/walls/$wallId/view"
                params={{ wallId: metadata.id }}
                className="flex flex-col items-center gap-3 p-6 bg-zinc-950 border border-zinc-800 rounded-lg 
                           hover:border-blue-500/50 hover:bg-blue-500/5 transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                  <Eye className="w-6 h-6 text-blue-400" />
                </div>
                <div className="text-center">
                  <div className="font-medium text-zinc-100">View Climbs</div>
                  <div className="text-xs text-zinc-500 mt-1">
                    Browse and explore existing climbs
                  </div>
                </div>
              </Link>

              <Link
                to="/walls/$wallId/create"
                params={{ wallId: metadata.id }}
                className="flex flex-col items-center gap-3 p-6 bg-zinc-950 border border-zinc-800 rounded-lg 
                           hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center group-hover:bg-emerald-500/20 transition-colors">
                  <Plus className="w-6 h-6 text-emerald-400" />
                </div>
                <div className="text-center">
                  <div className="font-medium text-zinc-100">Create Climb</div>
                  <div className="text-xs text-zinc-500 mt-1">
                    Set a new climb on this wall
                  </div>
                </div>
              </Link>

              <Link
                to="/walls/$wallId/holds"
                params={{ wallId: metadata.id }}
                className="flex flex-col items-center gap-3 p-6 bg-zinc-950 border border-zinc-800 rounded-lg 
                           hover:border-purple-500/50 hover:bg-purple-500/5 transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-purple-500/10 flex items-center justify-center group-hover:bg-purple-500/20 transition-colors">
                  <Pencil className="w-6 h-6 text-purple-400" />
                </div>
                <div className="text-center">
                  <div className="font-medium text-zinc-100">Edit Holds</div>
                  <div className="text-xs text-zinc-500 mt-1">
                    Modify hold positions and properties
                  </div>
                </div>
              </Link>
            </div>
          </div>

          {/* Danger Zone */}
          <div className="bg-zinc-900 border border-red-900/30 rounded-lg p-6">
            <h2 className="text-sm font-bold text-red-400 uppercase tracking-wider mb-4">
              Danger Zone
            </h2>

            {deleteError && (
              <div className="mb-4 px-4 py-3 bg-red-900/30 border border-red-800 rounded-lg text-red-300 text-sm">
                {deleteError}
              </div>
            )}

            {!showDeleteConfirm ? (
              <div className="flex items-center justify-between p-4 bg-zinc-950 border border-zinc-800 rounded-lg">
                <div>
                  <div className="font-medium text-zinc-100">
                    Delete this wall
                  </div>
                  <div className="text-sm text-zinc-500 mt-1">
                    Permanently remove this wall and all associated climbs. This
                    action cannot be undone.
                  </div>
                </div>
                <button
                  onClick={() => setShowDeleteConfirm(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600/10 border border-red-600/30 
                             text-red-400 rounded-lg hover:bg-red-600/20 hover:border-red-600/50 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete Wall
                </button>
              </div>
            ) : (
              <div className="p-4 bg-red-950/30 border border-red-800/50 rounded-lg">
                <div className="flex items-start gap-3 mb-4">
                  <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
                    <Trash2 className="w-5 h-5 text-red-400" />
                  </div>
                  <div>
                    <div className="font-medium text-zinc-100">
                      Are you sure you want to delete "{metadata.name}"?
                    </div>
                    <div className="text-sm text-zinc-400 mt-1">
                      This will permanently delete the wall, all{" "}
                      {holds?.length ?? 0} holds, and {metadata.num_climbs ?? 0}{" "}
                      climbs associated with it. This action cannot be undone.
                    </div>
                  </div>
                </div>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={() => setShowDeleteConfirm(false)}
                    disabled={isDeleting}
                    className="px-4 py-2 text-zinc-300 hover:text-zinc-100 transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleDelete}
                    disabled={isDeleting}
                    className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 
                               text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    {isDeleting ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Deleting...
                      </>
                    ) : (
                      <>
                        <Trash2 className="w-4 h-4" />
                        Yes, Delete Wall
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
