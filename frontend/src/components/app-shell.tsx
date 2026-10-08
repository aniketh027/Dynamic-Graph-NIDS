import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Bell, ChevronRight, CircleHelp, Cpu, Menu, Settings2, Shield, Unplug } from 'lucide-react'
import { Link, NavLink, Outlet, useLocation } from 'react-router'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { StatusBadge } from '@/components/status-badge'
import { navigation } from '@/lib/navigation'
import { projectFacts as facts } from '@/lib/project-facts'
import { useDemo } from '@/lib/demo-context'
import { IntrusionDrawer } from '@/components/intrusion-drawer'
import { ToastViewport } from '@/components/alert-card'
import { useBackend } from '@/lib/api/context'
import { adapterLabel, connectionLabel } from '@/lib/api/client'
import { useReplay, useReplayPresentation } from '@/lib/replay-context'

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const backend = useBackend()
  return (
    <>
      <Link to="/" className="brand" aria-label="Dynamic Graph NIDS dashboard" onClick={onNavigate}>
        <span className="brand-mark"><Shield size={22} strokeWidth={1.7} aria-hidden="true" /></span>
        <span className="sidebar-label"><strong>GRAPH<span>NIDS</span></strong><small>INTRUSION DETECTION</small></span>
      </Link>
      <div className="workspace-label sidebar-label"><span className="tiny-dot" />RESEARCH CONSOLE<span className="version-label">V1</span></div>
      <nav aria-label="Main navigation" className="sidebar-navigation">
        {['Overview', 'Workspace', 'Intelligence'].map(section => (
          <div className="nav-section" key={section}>
            <p className="nav-section-label sidebar-label">{section}</p>
            {navigation.filter(item => item.section === section).map(({ path, title, icon: Icon }) => (
              <NavLink key={path} to={path} end={path === '/'} title={title} aria-label={title} onClick={onNavigate}
                className={({ isActive }) => `nav-item${isActive ? ' nav-active' : ''}`}>
                <Icon size={17} strokeWidth={1.65} aria-hidden="true" /><span className="sidebar-label">{title}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="model-block">
          <div className="model-heading"><Cpu size={17} aria-hidden="true" /><span className="sidebar-label">MODEL CONFIGURATION</span></div>
          <dl className="sidebar-label"><div><dt>Model</dt><dd>{facts.model}</dd></div><div><dt>Mode</dt><dd>{facts.inferenceMode}</dd></div><div><dt>Console</dt><dd className="ready-label"><span className="tiny-dot" />Ready</dd></div></dl>
          <p className="sidebar-label">{connectionLabel(backend)}<br />{adapterLabel(backend)}</p>
        </div>
        <div className="sidebar-footnote sidebar-label"><Shield size={13} aria-hidden="true" /><span>Offline analysis workspace</span></div>
      </div>
    </>
  )
}

export function AppShell() {
  const demo = useDemo()
  const { state: replay } = useReplay()
  const presentation = useReplayPresentation()
  const backend = useBackend()
  const location = useLocation()
  const page = navigation.find(item => item.path === location.pathname)?.title ?? 'Page not found'
  const [mobileOpen, setMobileOpen] = useState(false)
  const contentRef = useRef<HTMLElement>(null)
  const lastPath = useRef(location.pathname)

  useEffect(() => {
    document.title = `${page} · GraphNIDS`
    if (lastPath.current !== location.pathname) {
      contentRef.current?.querySelector('h1')?.focus({ preventScroll: true })
      window.scrollTo({ top: 0, behavior: 'instant' })
      lastPath.current = location.pathname
    }
  }, [location.pathname, page])

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <aside className="sidebar"><SidebarContent /></aside>
      <div className="app-workspace">
        <header className="topbar">
          <div className="topbar-left">
            <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
              <SheetTrigger asChild><Button variant="ghost" size="icon" className="mobile-menu" aria-label="Open navigation"><Menu /></Button></SheetTrigger>
              <SheetContent side="left" className="mobile-sidebar">
                <SheetHeader className="sr-only"><SheetTitle>Main navigation</SheetTitle><SheetDescription>Navigate the intrusion detection console.</SheetDescription></SheetHeader>
                <SidebarContent onNavigate={() => setMobileOpen(false)} />
              </SheetContent>
            </Sheet>
            <span className="console-label">Console</span><ChevronRight size={13} className="breadcrumb-chevron" aria-hidden="true" /><span className="current-page">{page}</span>
          </div>
          <div className="topbar-actions">
            <span className={`environment-label${demo.enabled ? ' environment-demo' : ''}`}><span className="tiny-dot" />{demo.enabled ? `DEMO · ${demo.mode === 'running' ? 'SIMULATED' : demo.mode.toUpperCase()}` : replay.result ? `${presentation.badge} · ${presentation.qualifier}` : presentation.mock ? `${presentation.badge} · ${presentation.qualifier}` : replay.phase === 'analyzing' ? 'REAL ANALYSIS · PROCESSING' : 'FILE REPLAY'}</span>
            <span className="topbar-divider" />
            <Popover>
              <PopoverTrigger asChild><Button variant="ghost" size="icon" aria-label="Notifications"><Bell size={17} /></Button></PopoverTrigger>
              <PopoverContent align="end" className="header-popover"><h2>Notifications</h2><p>{demo.enabled ? 'A frontend demo is active. All displayed events, scores and timings are synthetic.' : replay.result ? `${replay.result.intrusion_prediction_count} intrusion predictions in the latest session analysis.` : 'No completed analysis in this session.'}</p><Link to="/intrusion-alerts" className="text-link">Open intrusion alerts<ChevronRight size={12} aria-hidden="true" /></Link></PopoverContent>
            </Popover>
            <Popover>
              <PopoverTrigger asChild><Button variant="ghost" size="icon" aria-label="System information"><Settings2 size={17} /></Button></PopoverTrigger>
              <PopoverContent align="end" className="header-popover"><h2>System information</h2><dl className="system-info"><div><dt>Workspace</dt><dd>Offline / file replay</dd></div><div><dt>Backend</dt><dd>{connectionLabel(backend)}</dd></div><div><dt>Adapter</dt><dd>{adapterLabel(backend)}</dd></div><div><dt>Traffic analysis</dt><dd>Bounded CSV replay</dd></div><div><dt>Live capture</dt><dd>Unsupported</dd></div><div><dt>Model family</dt><dd>HTGNN / GATv2</dd></div><div><dt>Input features</dt><dd>{facts.inputFeatures}</dd></div></dl><p>Demo mode uses synthetic frontend values.</p></PopoverContent>
            </Popover>
          </div>
        </header>
        <main id="main-content" ref={contentRef} className="main-content"><Outlet /></main>
        <footer className="app-footer"><span><Unplug size={12} aria-hidden="true" />{connectionLabel(backend)}<span className="footer-separator">/</span>Application preview</span><span><CircleHelp size={12} aria-hidden="true" />Research base: {facts.dataset}</span></footer>
      </div>
      <IntrusionDrawer />
      <ToastViewport />
    </div>
  )
}

export function PageHeading({ eyebrow, title, description, action }: {
  eyebrow: string; title: string; description: string; action?: ReactNode
}) {
  return <div className="page-heading"><div><p className="eyebrow">{eyebrow}</p><h1 tabIndex={-1}>{title}</h1><p className="page-description">{description}</p></div>{action}</div>
}

export function BackendNotice() {
  const backend = useBackend()
  return <div className="backend-notice" role="status"><Unplug size={14} aria-hidden="true" /><span><strong>{connectionLabel(backend)}.</strong> {adapterLabel(backend)}. Strict CSV replay is available in Analyze Traffic. Live capture is unsupported.</span><StatusBadge>File replay</StatusBadge></div>
}
