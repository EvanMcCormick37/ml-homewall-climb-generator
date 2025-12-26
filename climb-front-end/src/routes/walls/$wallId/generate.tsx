import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/walls/$wallId/generate')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/walls/$wallId/generate"!</div>
}
