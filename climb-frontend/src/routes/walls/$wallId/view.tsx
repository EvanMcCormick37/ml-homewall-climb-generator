import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/walls/$wallId/view')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/walls/$wallId/view"!</div>
}
