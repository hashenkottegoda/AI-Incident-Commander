import { Layout } from './components/Layout'
import { HealthCheckPage } from './pages/HealthCheckPage'

/**
 * Root component. No routing library yet -- this scaffolding step ships
 * exactly one real page (the backend health-check smoke test) inside the
 * shared Layout shell. Incident-list/detail/evaluation pages replace this
 * single-page wiring with real routes in later steps.
 */
function App() {
  return (
    <Layout>
      <HealthCheckPage />
    </Layout>
  )
}

export default App
