import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { getWall, getWallPhotoUrl } from "@/api/walls";
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

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate({ to: "/" })}
            className="text-zinc-400 hover:text-zinc-100 transition-colors"
          >
            ← Back
          </button>
          <h1 className="text-xl font-medium text-zinc-100">{metadata.name}</h1>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 p-6">
        <div className="max-w-4xl mx-auto">
          {/* Wall info card */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 mb-6">
            <div className="flex gap-6">
              {/* Wall photo */}
              <div className="w-48 h-48 bg-zinc-800 rounded overflow-hidden flex-shrink-0">
                <img
                  src={getWallPhotoUrl(metadata.id)}
                  alt={metadata.name}
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Wall details */}
              <div className="flex-1">
                <h2 className="text-lg font-medium text-zinc-100 mb-4">
                  Wall Details
                </h2>
                <dl className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <dt className="text-zinc-500">Holds</dt>
                    <dd className="text-zinc-100">{holds?.length ?? 0}</dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Climbs</dt>
                    <dd className="text-zinc-100">{metadata.num_climbs ?? 0}</dd>
                  </div>
                  {metadata.dimensions && (
                    <div>
                      <dt className="text-zinc-500">Dimensions</dt>
                      <dd className="text-zinc-100">
                        {metadata.dimensions[0]} × {metadata.dimensions[1]} cm
                      </dd>
                    </div>
                  )}
                  {metadata.angle !== undefined && metadata.angle !== null && (
                    <div>
                      <dt className="text-zinc-500">Angle</dt>
                      <dd className="text-zinc-100">{metadata.angle}°</dd>
                    </div>
                  )}
                </dl>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-4">
            <Link
              to="/walls/$wallId/holds"
              params={{ wallId: metadata.id }}
              className="px-4 py-2 bg-zinc-800 text-zinc-100 hover:bg-zinc-700 transition-colors rounded"
            >
              Edit Holds
            </Link>
            {/* Add more actions here as needed */}
          </div>
        </div>
      </div>
    </div>
  );
}
