// routes/sign-in.tsx
import { createFileRoute } from "@tanstack/react-router";
import { SignIn } from "@clerk/clerk-react";

export const Route = createFileRoute("/signIn")({
  component: () => (
    <div
      style={{ display: "flex", justifyContent: "center", paddingTop: "80px" }}
    >
      <SignIn routing="hash" />
    </div>
  ),
});
