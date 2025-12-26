import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/walls/')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/walls/"!</div>
}
