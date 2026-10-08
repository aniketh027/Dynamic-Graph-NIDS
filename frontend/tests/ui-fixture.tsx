/** Development-only visual test entry; never imported by the application build.
 * Every attack below is a MOCKED UI VALIDATION fixture, not model output. */
import { createRoot } from 'react-dom/client'
import { useState, useSyncExternalStore } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { LazyMotion, domAnimation, MotionConfig } from 'motion/react'
import { AppShell } from '@/components/app-shell'
import { DemoProvider } from '@/components/demo-provider'
import { ReplayContext } from '@/lib/replay-context'
import { BackendContext } from '@/lib/api/context'
import { ReplaySession } from '@/lib/replay-session'
import { Dashboard } from '@/pages/dashboard'
import { AnalyzeTraffic, IntrusionAlerts, NetworkMonitor, ModelPerformance, GraphOptimization } from '@/pages/workspace-pages'
import wire from './synthetic-wire-response.json'
import type { AnalysisResponse, ValidationResponse } from '@/lib/api/replay'
import '@/index.css'
import '@/styles/replay.css'

const mock = structuredClone(wire.analysis) as AnalysisResponse
mock.source_filename = 'MOCKED-UI-VALIDATION.csv'
mock.analysis_id = 'MOCKED-UI-VALIDATION'
mock.provenance.analysis_id = mock.analysis_id
mock.input_flow_count = 525; mock.complete_window_count = 105; mock.trailing_flow_count = 0
mock.benign_prediction_count = 0; mock.intrusion_prediction_count = 105
mock.prediction_class_counts = Object.fromEntries(Object.keys(mock.prediction_class_counts).map(k => [k,k === 'Bot' ? 105 : 0]))
mock.predictions = Array.from({length:105}, (_,i) => ({window_index:i,target_flow_index:i*5+4,prediction_class_id:1,prediction_label:'Bot',model_score:0.75,is_attack:true}))
const file = new File([' '.repeat(mock.source_size_bytes)],mock.source_filename)
const validation = {...wire.validation,filename:file.name,flow_row_count:525,complete_window_count:105,trailing_flow_count:0} as ValidationResponse
export function Fixture() {
  const [session] = useState(() => new ReplaySession('http://unused.invalid', {
    validateReplay:async()=>{await new Promise(r=>setTimeout(r,3000));return validation},
    analyzeReplay:async()=>{await new Promise(r=>setTimeout(r,8000));return mock},
  }))
  const state=useSyncExternalStore(session.subscribe,session.getSnapshot)
  return <MotionConfig reducedMotion="user"><LazyMotion features={domAnimation}><BackendContext.Provider value={{state:'connected',status:{backend_connected:true,model_adapter_state:'ready',model_loaded:true,inference_ready:true,live_capture_supported:false,file_replay_supported:true,mode:'offline_application_preview',error:null}}}><DemoProvider><ReplayContext.Provider value={{session,state,presentation:'mock'}}>
    <div style={{position:'fixed',bottom:0,left:0,right:0,zIndex:9999,padding:10,background:'#63242b',color:'white',fontSize:13}}>MOCKED UI VALIDATION · No model output. <button onClick={()=>{session.select(file);void session.validate()}}>Load mocked validation</button></div>
    <MemoryRouter initialEntries={['/analyze-traffic']}><Routes><Route element={<AppShell />}><Route index element={<Dashboard />} /><Route path="analyze-traffic" element={<AnalyzeTraffic />} /><Route path="intrusion-alerts" element={<IntrusionAlerts />} /><Route path="network-monitor" element={<NetworkMonitor />} /><Route path="model-performance" element={<ModelPerformance />} /><Route path="graph-optimization" element={<GraphOptimization />} /></Route></Routes></MemoryRouter>
  </ReplayContext.Provider></DemoProvider></BackendContext.Provider></LazyMotion></MotionConfig>
}
createRoot(document.getElementById('root')!).render(<Fixture />)
