import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
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

function HomePage() {
  const { walls, loading, error } = useWalls();
  const navigate = useNavigate();

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col">
      {/* Welcome section */}
      <section className="px-6 py-12 text-center">
        <p className="text-zinc-400 max-w-2xl mx-auto">
          Welcome to BetaZero, a public resource for generating board climbs using ML techniques.
        </p>
      </section>

      {/* Main content area */}
      <div className="flex-1 flex items-start justify-center px-6 pb-24">
        <div className="flex gap-24 items-start">
          {/* Wall selection menu */}
          <div className="min-w-[280px]">
            <div className="border border-zinc-700 rounded overflow-hidden">
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

              {/* Existing walls */}
              {!loading && walls.map((wall) => (
                <button
                  key={wall.id}
                  onClick={() => navigate({ to: `/walls/${wall.id}` })}
                  className="w-full px-4 py-3 text-left text-zinc-100 hover:bg-zinc-800 transition-colors border-b border-zinc-700 last:border-b-0"
                >
                  {wall.name}
                </button>
              ))}

              {/* Show placeholder if no walls exist */}
              {!loading && walls.length === 0 && (
                <button
                  disabled
                  className="w-full px-4 py-3 text-left text-zinc-500 border-b border-zinc-700"
                >
                  No walls yet
                </button>
              )}

              {/* Coming soon walls */}
              {COMING_SOON_WALLS.map((wall) => (
                <button
                  key={wall.id}
                  disabled
                  className="w-full px-4 py-3 text-left text-zinc-600 border-b border-zinc-700 cursor-not-allowed"
                >
                  <span className="text-zinc-700">(Coming soon)</span> {wall.name}
                </button>
              ))}

              {/* Create new wall link */}
              <Link
                to="/walls/new"
                className="block w-full px-4 py-3 text-left text-purple-400 hover:bg-zinc-800/50 transition-colors"
              >
                Create new wall
              </Link>
            </div>
          </div>

          {/* Side links */}
          <div className="flex flex-col gap-3 min-w-[160px]">
            <a
              href="#"
              className="block px-4 py-2 border border-zinc-700 rounded text-zinc-300 hover:border-zinc-500 hover:text-zinc-100 transition-colors text-sm"
            >
              About me
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="block px-4 py-2 border border-zinc-700 rounded text-zinc-300 hover:border-zinc-500 hover:text-zinc-100 transition-colors text-sm"
            >
              Github Repo
            </a>
            <a
              href="#"
              className="block px-4 py-2 border border-zinc-700 rounded text-zinc-300 hover:border-zinc-500 hover:text-zinc-100 transition-colors text-sm"
            >
              Write-up (Substack)
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
