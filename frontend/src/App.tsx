import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HealthCheckPage } from './pages/HealthCheckPage'
import { IncidentDetailPage } from './pages/IncidentDetailPage'
import { IncidentListPage } from './pages/IncidentListPage'

/**
 * Root component. `/` is the incident list (the dashboard's landing page);
 * `/incidents/:id` is the detail view (`GET /{incident_id}`); `/health`
 * keeps the original backend-connectivity smoke test reachable as its own
 * route now that it's no longer the only page.
 */
function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<IncidentListPage />} />
          <Route path="/incidents/:id" element={<IncidentDetailPage />} />
          <Route path="/health" element={<HealthCheckPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
