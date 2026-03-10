// __root.tsx
import { ClerkProvider, useAuth } from "@clerk/clerk-react";
import { Outlet, createRootRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { setAuthTokenProvider } from "@/api/client";

function AuthInit() {
  const { getToken } = useAuth();

  useEffect(() => {
    setAuthTokenProvider(() => getToken());
  }, [getToken]);

  return null;
}

const CLERK_PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <ClerkProvider publishableKey={CLERK_PUBLISHABLE_KEY}>
      <AuthInit />
      <Outlet />
    </ClerkProvider>
  );
}
