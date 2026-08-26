import type { ReactNode } from 'react'

const NAV_ITEMS = [
  { label: 'Incidents', href: '#', disabled: false },
  { label: 'Evaluation', href: '#', disabled: true },
] as const

/**
 * Root page shell: a header/nav bar plus a content area. Deliberately
 * minimal at this scaffolding step -- the incident-list/detail/evaluation
 * pages slot into `children` in later steps, most of them behind real
 * client-side routing (not built yet, so nav links are placeholders).
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
              <span
                key={item.label}
                className={
                  item.disabled
                    ? 'cursor-not-allowed text-slate-600'
                    : 'text-slate-300 hover:text-slate-50'
                }
                title={item.disabled ? 'Not built yet' : undefined}
              >
                {item.label}
              </span>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>
    </div>
  )
}
