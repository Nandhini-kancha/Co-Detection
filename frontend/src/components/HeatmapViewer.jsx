import React, { useState } from 'react'
import { Eye, Layers, Maximize2, Minimize2 } from 'lucide-react'

export default function HeatmapViewer({ heatmapBase64, originalImage, primaryDiagnosis }) {
  const [viewMode, setViewMode] = useState('side-by-side') // 'original', 'heatmap', 'side-by-side'
  const [fullscreen, setFullscreen] = useState(false)

  const heatmapSrc = `data:image/png;base64,${heatmapBase64}`

  return (
    <div className={`glass-card p-6 animate-slide-up ${
      fullscreen ? 'fixed inset-4 z-50 overflow-auto' : ''
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold text-dark-200 uppercase tracking-wider flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary-400" />
          Grad-CAM Visualization
        </h2>
        <div className="flex items-center gap-2">
          {/* View toggles */}
          {['original', 'heatmap', 'side-by-side'].map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                viewMode === mode
                  ? 'bg-primary-500/20 text-primary-300 border border-primary-500/30'
                  : 'text-dark-400 hover:text-dark-200 hover:bg-dark-800/50'
              }`}
            >
              {mode === 'side-by-side' ? 'Compare' : mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-1.5 rounded-lg text-dark-400 hover:text-dark-200 hover:bg-dark-800/50 transition-all ml-1"
          >
            {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Image Display */}
      <div className={`grid gap-4 ${
        viewMode === 'side-by-side' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
      }`}>
        {(viewMode === 'original' || viewMode === 'side-by-side') && (
          <div className="space-y-2">
            <p className="text-xs text-dark-400 font-medium">Original X-Ray</p>
            <div className="rounded-xl overflow-hidden bg-dark-900/50 border border-dark-700/30">
              <img
                src={originalImage}
                alt="Original chest X-ray"
                className="w-full h-auto object-contain max-h-[400px]"
              />
            </div>
          </div>
        )}

        {(viewMode === 'heatmap' || viewMode === 'side-by-side') && (
          <div className="space-y-2">
            <p className="text-xs text-dark-400 font-medium">Grad-CAM Heatmap — {primaryDiagnosis}</p>
            <div className="rounded-xl overflow-hidden bg-dark-900/50 border border-dark-700/30 relative">
              <img
                src={heatmapSrc}
                alt="Grad-CAM heatmap overlay"
                className="w-full h-auto object-contain max-h-[400px]"
              />
              <div className="absolute bottom-3 right-3 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-dark-900/80 backdrop-blur-sm border border-dark-700/50">
                <Eye className="w-3 h-3 text-primary-400" />
                <span className="text-[11px] text-dark-300 font-medium">AI Focus Region</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-6 px-1">
        <div className="flex items-center gap-2">
          <div className="w-16 h-2 rounded-full" style={{ background: 'linear-gradient(to right, #0000ff, #00ff00, #ffff00, #ff0000)' }} />
          <span className="text-[11px] text-dark-500">Low → High Activation</span>
        </div>
        <p className="text-[11px] text-dark-600">
          Highlighted regions indicate areas the model focused on for its prediction.
        </p>
      </div>
    </div>
  )
}
