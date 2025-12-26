import { createRootRoute, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: () => (
    <div>
      {/* Your layout/nav goes here */}
      <Outlet />
    </div>
  ),
});
