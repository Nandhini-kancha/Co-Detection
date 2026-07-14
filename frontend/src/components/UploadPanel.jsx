import React, { useState, useRef } from 'react'
import { Upload, X, Image as ImageIcon, Scan, RotateCcw } from 'lucide-react'

export default function UploadPanel({ onFileSelect, onAnalyze, onReset, selectedFile, previewUrl, loading, error }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDragIn = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragOut = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      onFileSelect(files[0])
    }
  }

  const handleChange = (e) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onFileSelect(files[0])
    }
  }

  return (
    <div className="glass-card p-6 space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-dark-200 uppercase tracking-wider flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-primary-400" />
          X-Ray Upload
        </h2>
        {selectedFile && (
          <button
            onClick={onReset}
            className="text-dark-400 hover:text-dark-200 transition-colors p-1 rounded-lg hover:bg-dark-700/50"
            title="Reset"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        )}
      </div>

      {!selectedFile ? (
        <div
          className={`upload-zone rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer min-h-[280px] transition-all ${
            isDragging ? 'drag-over' : ''
          }`}
          onClick={() => inputRef.current?.click()}
          onDragOver={handleDrag}
          onDragEnter={handleDragIn}
          onDragLeave={handleDragOut}
          onDrop={handleDrop}
        >
          <div className="w-14 h-14 rounded-2xl bg-primary-500/10 flex items-center justify-center mb-4">
            <Upload className={`w-7 h-7 text-primary-400 transition-transform ${isDragging ? 'scale-110' : ''}`} />
          </div>
          <p className="text-dark-200 font-medium text-sm mb-1">Drop X-ray image here</p>
          <p className="text-dark-500 text-xs">or click to browse</p>
          <p className="text-dark-600 text-[11px] mt-3">Supports PNG, JPG, DICOM</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
            id="xray-upload"
          />
        </div>
      ) : (
        <div className="space-y-4">
          {/* Preview */}
          <div className="relative rounded-xl overflow-hidden bg-dark-900/50 border border-dark-700/30">
            <img
              src={previewUrl}
              alt="X-ray preview"
              className="w-full h-64 object-contain"
            />
            <button
              onClick={onReset}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-dark-900/80 hover:bg-dark-800 text-dark-300 hover:text-white transition-all"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* File info */}
          <div className="flex items-center justify-between text-xs px-1">
            <span className="text-dark-400 truncate max-w-[60%]">{selectedFile.name}</span>
            <span className="text-dark-500">{(selectedFile.size / 1024).toFixed(1)} KB</span>
          </div>

          {/* Analyze button */}
          <button
            onClick={onAnalyze}
            disabled={loading}
            className={`w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
              loading
                ? 'bg-dark-700 text-dark-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-400 hover:to-primary-500 text-white shadow-lg shadow-primary-500/25 hover:shadow-primary-500/40 active:scale-[0.98]'
            }`}
            id="analyze-btn"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-dark-500 border-t-primary-400 rounded-full animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Scan className="w-4 h-4" />
                Analyze X-Ray
              </>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
