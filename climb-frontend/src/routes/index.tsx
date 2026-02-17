import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useWalls } from "@/hooks/useWalls";
import { getWallPhotoUrl } from "@/api/walls";

export const Route = createFileRoute("/")({
  component: HomePage,
});

// External links
const LINKS = [
  { label: "About me", href: "https://www.evmojo.dev" },
  {
    label: "Github Repo",
    href: "https://github.com/EvanMcCormick37/ml-homewall-climb-generator",
  },
  {
    label: "Write-up (Substack)",
    href: "https://evmojo37.substack.com/p/beta-zero-alpha-can-ai-set-climbs",
  },
];

function HomePage() {
  const { walls, loading, error } = useWalls();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Main content area */}
      <div className="flex-1 flex flex-col items-center px-6 py-12">
        {/* Top section with welcome text and links */}
        <div className="w-full max-w-4xl flex items-start justify-between mb-16">
          {/* Welcome text */}
          <p className="text-zinc-400 max-w-md">
            Welcome to BetaZero, a public resource for generating board climbs
            using ML techniques.
          </p>

          {/* External links */}
          <div className="flex flex-col gap-2">
            {LINKS.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-zinc-900 text-zinc-100 text-sm hover:bg-zinc-800 transition-colors border border-zinc-800"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>

        {/* Wall list section */}
        <div className="w-full max-w-4xl">
          <h2 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-6">
            Select a Wall
          </h2>

          {/* Loading state */}
          {loading && (
            <div className="text-center text-zinc-500 py-12">
              Loading walls...
            </div>
          )}

          {/* Error state */}
          {error && (
            <div className="text-center text-red-400 py-12">{error}</div>
          )}

          {/* Wall cards */}
          {!loading && !error && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Available walls */}
              {walls.map((wall) => (
                <button
                  key={wall.id}
                  onClick={() =>
                    navigate({ to: "/$wallId", params: { wallId: wall.id } })
                  }
                  className="group text-left bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden 
                             hover:border-cyan-500/50 hover:bg-zinc-900/80 transition-all"
                >
                  {/* Wall photo */}
                  <div className="w-full h-40 bg-zinc-800 overflow-hidden">
                    <img
                      src={getWallPhotoUrl(wall.id)}
                      alt={wall.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  </div>
                  {/* Wall info */}
                  <div className="p-4">
                    <div className="font-medium text-zinc-100">{wall.name}</div>
                    <div className="text-xs text-zinc-500 mt-1">
                      {wall.num_holds} holds
                      {wall.dimensions &&
                        ` · ${wall.dimensions[0]}×${wall.dimensions[1]} ft`}
                      {wall.angle != null && ` · ${wall.angle}°`}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
