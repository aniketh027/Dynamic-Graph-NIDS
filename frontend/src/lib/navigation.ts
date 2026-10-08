import {
  Activity, ChartNoAxesCombined, FileUp, LayoutDashboard,
  Network, ShieldAlert,
} from 'lucide-react'

export const navigation = [
  { path: '/', title: 'Dashboard', icon: LayoutDashboard, section: 'Overview' },
  { path: '/network-monitor', title: 'Network Monitor', icon: Activity, section: 'Workspace' },
  { path: '/analyze-traffic', title: 'Analyze Traffic', icon: FileUp, section: 'Workspace' },
  { path: '/intrusion-alerts', title: 'Intrusion Alerts', icon: ShieldAlert, section: 'Workspace' },
  { path: '/graph-optimization', title: 'Graph Optimization', icon: Network, section: 'Intelligence' },
  { path: '/model-performance', title: 'Model Performance', icon: ChartNoAxesCombined, section: 'Intelligence' },
] as const
