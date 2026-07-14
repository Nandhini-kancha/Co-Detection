import React, { useState, useEffect, useCallback } from 'react'
import UploadPanel from './components/UploadPanel'
import ResultCard from './components/ResultCard'
import HeatmapViewer from './components/HeatmapViewer'
import { Activity, Shield, AlertTriangle, Zap, Heart, Cpu } from 'lucide-react'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiHealth, setApiHealth] = useState(null)
  const [analysisHistory, setAnalysisHistory] = useState([])

  useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      const res = await fetch('/health')
      const data = await res.json()
      setApiHealth(data)
    } catch (err) {
      setApiHealth({ status: 'offline' })
    }
  }

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResults(null)
    setError(null)
  }, [])

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Analysis failed')
      }

      const data = await res.json()
      setResults(data)
      setAnalysisHistory(prev => [{
        id: Date.now(),
        filename: selectedFile.name,
        diagnosis: data.primary_diagnosis,
        severity: data.severity.level,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev].slice(0, 10))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResults(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-dark-950 relative overflow-hidden">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary-900/3 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-dark-700/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
                  <Activity className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-emerald-400 rounded-full border-2 border-dark-950 animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-white to-dark-300 bg-clip-text text-transparent">
                  ChestAI Screening
                </h1>
                <p className="text-xs text-dark-400 font-medium tracking-wide uppercase">
                  Pneumonia & Tuberculosis Co-Detection
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              {/* Model info badge */}
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full glass-card-lighter">
                <Cpu className="w-3.5 h-3.5 text-primary-400" />
                <span className="text-xs text-dark-300 font-medium">DenseNet-121</span>
              </div>

              {/* Health status */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                apiHealth?.status === 'healthy'
                  ? 'bg-emerald-500/10 border border-emerald-500/20'
                  : apiHealth?.status === 'offline'
                  ? 'bg-red-500/10 border border-red-500/20'
                  : 'bg-amber-500/10 border border-amber-500/20'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  apiHealth?.status === 'healthy' ? 'bg-emerald-400 animate-pulse' :
                  apiHealth?.status === 'offline' ? 'bg-red-400' : 'bg-amber-400 animate-pulse'
                }`} />
                <span className="text-xs font-medium">
                  {apiHealth?.status === 'healthy' ? 'API Online' :
                   apiHealth?.status === 'offline' ? 'API Offline' : 'Connecting...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Top Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { icon: Shield, label: 'Detection Model', value: 'Multi-Label CNN', color: 'text-primary-400' },
            { icon: Zap, label: 'Inference', value: results ? `${results.inference_time_seconds}s` : '—', color: 'text-amber-400' },
            { icon: Heart, label: 'Conditions', value: 'Pneumonia · TB', color: 'text-rose-400' },
            { icon: AlertTriangle, label: 'Severity', value: results?.severity?.level || '—', color: results?.severity?.level === 'Severe' ? 'text-red-400' : results?.severity?.level === 'Moderate' ? 'text-orange-400' : results?.severity?.level === 'Mild' ? 'text-amber-400' : 'text-emerald-400' },
          ].map((stat, i) => (
            <div key={i} className="glass-card-lighter p-4 flex items-center gap-3">
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
              <div>
                <p className="text-[11px] text-dark-400 uppercase tracking-wider font-medium">{stat.label}</p>
                <p className="text-sm font-semibold text-dark-100">{stat.value}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column - Upload */}
          <div className="lg:col-span-4">
            <UploadPanel
              onFileSelect={handleFileSelect}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              loading={loading}
              error={error}
            />
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-8 space-y-6">
            {loading && (
              <div className="glass-card p-8 animate-fade-in">
                <div className="flex flex-col items-center gap-4">
                  <div className="relative">
                    <div className="w-16 h-16 rounded-full border-4 border-dark-700 border-t-primary-400 animate-spin" />
                    <div className="absolute inset-0 w-16 h-16 rounded-full border-4 border-transparent border-b-purple-400 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
                  </div>
                  <div className="text-center">
                    <p className="text-dark-200 font-semibold">Analyzing X-ray...</p>
                    <p className="text-dark-400 text-sm mt-1">Running DenseNet-121 inference & Grad-CAM</p>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="glass-card p-6 border-red-500/30 animate-fade-in">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
                  <div>
                    <p className="text-red-300 font-semibold">Analysis Failed</p>
                    <p className="text-red-400/70 text-sm mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {results && (
              <>
                <ResultCard results={results} />
                <HeatmapViewer
                  heatmapBase64={results.heatmap_base64}
                  originalImage={previewUrl}
                  primaryDiagnosis={results.primary_diagnosis}
                />
              </>
            )}

            {!loading && !results && !error && (
              <div className="glass-card p-12 flex flex-col items-center justify-center text-center animate-fade-in">
                <div className="w-20 h-20 rounded-2xl bg-dark-800/50 flex items-center justify-center mb-6">
                  <Activity className="w-10 h-10 text-dark-500" />
                </div>
                <h3 className="text-lg font-semibold text-dark-300 mb-2">No Analysis Yet</h3>
                <p className="text-dark-500 text-sm max-w-md">
                  Upload a chest X-ray image and click "Analyze" to get AI-powered
                  Pneumonia & Tuberculosis screening results with Grad-CAM heatmaps.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Analysis History */}
        {analysisHistory.length > 0 && (
          <div className="mt-8 glass-card p-6 animate-slide-up">
            <h3 className="text-sm font-semibold text-dark-300 uppercase tracking-wider mb-4">Recent Analyses</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-dark-700/50">
                    <th className="text-left text-dark-400 font-medium pb-3 pr-4">Time</th>
                    <th className="text-left text-dark-400 font-medium pb-3 pr-4">File</th>
                    <th className="text-left text-dark-400 font-medium pb-3 pr-4">Diagnosis</th>
                    <th className="text-left text-dark-400 font-medium pb-3">Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {analysisHistory.map((item) => (
                    <tr key={item.id} className="border-b border-dark-800/50 hover:bg-dark-800/30 transition-colors">
                      <td className="py-2.5 pr-4 text-dark-400 font-mono text-xs">{item.timestamp}</td>
                      <td className="py-2.5 pr-4 text-dark-200">{item.filename}</td>
                      <td className="py-2.5 pr-4">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          item.diagnosis === 'Normal' ? 'bg-emerald-500/10 text-emerald-400' :
                          item.diagnosis === 'Pneumonia' ? 'bg-amber-500/10 text-amber-400' :
                          'bg-red-500/10 text-red-400'
                        }`}>{item.diagnosis}</span>
                      </td>
                      <td className="py-2.5">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          item.severity === 'Normal' ? 'bg-emerald-500/10 text-emerald-400' :
                          item.severity === 'Mild' ? 'bg-amber-500/10 text-amber-400' :
                          item.severity === 'Moderate' ? 'bg-orange-500/10 text-orange-400' :
                          'bg-red-500/10 text-red-400'
                        }`}>{item.severity}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-dark-800/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-xs text-dark-500">
              ⚕️ <span className="text-dark-400 font-medium">Clinical Disclaimer:</span> This tool is for research and screening purposes only.
              Results must be validated by a qualified radiologist. Not approved for clinical diagnosis.
            </p>
            <p className="text-xs text-dark-600">ChestAI v1.0 · DenseNet-121 · Grad-CAM</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
