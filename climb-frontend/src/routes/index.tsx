import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useWalls } from "@/hooks/useWalls";

export const Route = createFileRoute("/")({
  component: HomePage,
});

// Coming soon walls (hardcoded for now)
const COMING_SOON_WALLS = [
  { name: "Kilter Board", id: "kilter" },
  { name: "Tension Board 2", id: "tension" },
  { name: "Decoy Board", id: "decoy" },
];

// External links
const LINKS = [
  { label: "About me", href: "https://www.evmojo.dev" },
  { label: "Github Repo", href: "https://github.com/EvanMcCormick37/ml-homewall-climb-generator" },
  { label: "Write-up (Substack)", href: "https://evmojo37.substack.com/p/beta-zero-alpha-can-ai-set-climbs" },
];

function HomePage() {
  const { walls, loading, error } = useWalls();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleWallSelect = (wallId: string) => {
    setIsMenuOpen(false);
    navigate({ to: `/walls/${wallId}` });
  };

  const handleCreateWall = () => {
    setIsMenuOpen(false);
    navigate({ to: "/walls/new" });
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Main content area */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        {/* Top section with welcome text and links */}
        <div className="w-full max-w-4xl flex items-start justify-between mb-24">
          {/* Welcome text */}
          <p className="text-zinc-400 max-w-md">
            Welcome to BetaZero, a public resource for generating board climbs using ML techniques.
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

        {/* Center section with Select Wall button */}
        <div className="relative">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="px-8 py-3 bg-zinc-900 text-zinc-100 hover:bg-zinc-800 transition-colors border border-zinc-700"
          >
            Select Wall
          </button>

          {/* Wall selection dropdown */}
          {isMenuOpen && (
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 bg-zinc-900 border border-zinc-700 rounded shadow-xl z-10">
              {/* Loading state */}
              {loading && (
                <div className="px-4 py-3 text-center text-zinc-500 text-sm">
                  Loading walls...
                </div>
              )}

              {/* Error state */}
              {error && (
                <div className="px-4 py-3 text-center text-red-400 text-sm">
                  {error}
                </div>
              )}

              {/* Available walls */}
              {!loading && walls.map((wall) => (
                <button
                  key={wall.id}
                  onClick={() => handleWallSelect(wall.id)}
                  className="w-full px-4 py-2 text-center text-zinc-100 hover:bg-zinc-800 transition-colors border-b border-zinc-700 last:border-b-0"
                >
                  {wall.name}
                </button>
              ))}

              {/* Coming soon walls */}
              {!loading && COMING_SOON_WALLS.map((wall) => (
                <div
                  key={wall.id}
                  className="w-full px-4 py-2 text-center text-zinc-500 border-b border-zinc-700 cursor-not-allowed"
                >
                  (Coming soon) {wall.name}
                </div>
              ))}

              {/* Create new wall option */}
              {!loading && (
                <button
                  onClick={handleCreateWall}
                  className="w-full px-4 py-2 text-center text-blue-400 hover:bg-zinc-800 transition-colors"
                >
                  Create new wall
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
