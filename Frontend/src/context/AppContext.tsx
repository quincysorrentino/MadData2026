import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import type { BoundingBox } from '../types'

interface AppState {
  selectedBodyPartId: number | null
  selectedBodyPartName: string | null
  uploadedImageFile: File | null
  uploadedImageObjectUrl: string | null
  diagnosis: string | null
  boundingBox: BoundingBox | null
}

interface AppContextValue extends AppState {
  setBodyPart: (id: number, name: string) => void
  setUploadedImage: (file: File) => void
  setDiagnosisResult: (diagnosis: string, box: BoundingBox | null) => void
  resetSession: () => void
}

const defaultState: AppState = {
  selectedBodyPartId: null,
  selectedBodyPartName: null,
  uploadedImageFile: null,
  uploadedImageObjectUrl: null,
  diagnosis: null,
  boundingBox: null,
}

const AppContext = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>(defaultState)

  const setBodyPart = useCallback((id: number, name: string) => {
    setState(prev => ({ ...prev, selectedBodyPartId: id, selectedBodyPartName: name }))
  }, [])

  const setUploadedImage = useCallback((file: File) => {
    setState(prev => {
      if (prev.uploadedImageObjectUrl) {
        URL.revokeObjectURL(prev.uploadedImageObjectUrl)
      }
      const url = URL.createObjectURL(file)
      return { ...prev, uploadedImageFile: file, uploadedImageObjectUrl: url }
    })
  }, [])

  const setDiagnosisResult = useCallback((diagnosis: string, box: BoundingBox | null) => {
    setState(prev => ({ ...prev, diagnosis, boundingBox: box }))
  }, [])

  const resetSession = useCallback(() => {
    setState(prev => {
      if (prev.uploadedImageObjectUrl) {
        URL.revokeObjectURL(prev.uploadedImageObjectUrl)
      }
      return defaultState
    })
  }, [])

  return (
    <AppContext.Provider
      value={{
        ...state,
        setBodyPart,
        setUploadedImage,
        setDiagnosisResult,
        resetSession,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
