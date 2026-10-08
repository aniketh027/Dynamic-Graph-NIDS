import { domAnimation, LazyMotion, MotionConfig } from 'motion/react'
import { BrowserRouter, Route, Routes } from 'react-router'
import { AppShell } from '@/components/app-shell'
import { DemoProvider } from '@/components/demo-provider'
import { BackendProvider } from '@/components/backend-provider'
import { ReplayProvider } from '@/components/replay-provider'
import '@/styles/replay.css'
import { Dashboard } from '@/pages/dashboard'
import { AnalyzeTraffic, GraphOptimization, IntrusionAlerts, ModelPerformance, NetworkMonitor, NotFound } from '@/pages/workspace-pages'

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <LazyMotion features={domAnimation}>
        <BackendProvider><DemoProvider><ReplayProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<AppShell />}>
                <Route index element={<Dashboard />} />
                <Route path="network-monitor" element={<NetworkMonitor />} />
                <Route path="analyze-traffic" element={<AnalyzeTraffic />} />
                <Route path="intrusion-alerts" element={<IntrusionAlerts />} />
                <Route path="graph-optimization" element={<GraphOptimization />} />
                <Route path="model-performance" element={<ModelPerformance />} />
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ReplayProvider></DemoProvider></BackendProvider>
      </LazyMotion>
    </MotionConfig>
  )
}
