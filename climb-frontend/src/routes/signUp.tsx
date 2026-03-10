import { createFileRoute } from "@tanstack/react-router";
import { SignUp } from "@clerk/clerk-react";

export const Route = createFileRoute("/signUp")({
  component: () => (
    <div
      style={{ display: "flex", justifyContent: "center", paddingTop: "80px" }}
    >
      <SignUp routing="hash" />
    </div>
  ),
});
