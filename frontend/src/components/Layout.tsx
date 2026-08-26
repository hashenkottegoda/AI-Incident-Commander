import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { label: 'Incidents', to: '/' },
  { label: 'Evaluation', to: '/evaluation' },
  { label: 'Health', to: '/health' },
] as const

/**
 * Root page shell: a header/nav bar plus a content area. `App.tsx` renders
 * `<Layout><Routes>...</Routes></Layout>` directly (plain `children`, not
 * `react-router-dom`'s `<Outlet />` -- there's no nested-route layout here
 * to warrant it).
 */
export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-semibold tracking-tight text-slate-50">
              AI Incident Commander
            </span>
            <span className="text-xs text-slate-500">investigation dashboard</span>
          </div>
          <nav className="flex gap-4 text-sm">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.label}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  isActive ? 'text-slate-50' : 'text-slate-300 hover:text-slate-50'
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>
    </div>
  )
}
