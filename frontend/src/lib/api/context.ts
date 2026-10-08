import { createContext, useContext } from 'react'
import type { Connection } from './client'

export const BackendContext = createContext<Connection>({ state: 'checking', status: null })
export const useBackend = () => useContext(BackendContext)
