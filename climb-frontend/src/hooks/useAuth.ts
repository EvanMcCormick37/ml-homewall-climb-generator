// hooks/useAuth.ts
import { useUser, useAuth } from "@clerk/clerk-react";

export function useBetaZeroAuth() {
  const { user, isLoaded, isSignedIn } = useUser();
  const { getToken } = useAuth();

  return {
    user,
    isLoaded,
    isSignedIn,
    userId: user?.id ?? null,
    displayName: user?.fullName ?? user?.username ?? null,
    email: user?.primaryEmailAddress?.emailAddress ?? null,
    getApiToken: () => getToken(),
  };
}
