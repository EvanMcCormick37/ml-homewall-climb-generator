import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/walls/$wallId/create')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/walls/$wallId/create"!</div>
}
