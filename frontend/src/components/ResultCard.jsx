import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { AlertCircle, CheckCircle, TrendingUp, Clock } from 'lucide-react'

const COLORS = {
  Pneumonia: '#f59e0b',
  Tuberculosis: '#ef4444',
  Normal: '#10b981',
}

const SEVERITY_STYLES = {
  Normal: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', glow: 'shadow-emerald-500/20' },
  Mild: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', glow: 'shadow-amber-500/20' },
  Moderate: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', glow: 'shadow-orange-500/20' },
  Severe: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', glow: 'shadow-red-500/20' },
}

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card-lighter px-3 py-2 shadow-xl">
        <p className="text-xs font-semibold text-dark-100">{payload[0].payload.name}</p>
        <p className="text-xs text-dark-300">{(payload[0].value * 100).toFixed(1)}%</p>
      </div>
    )
  }
  return null
}

export default function ResultCard({ results }) {
  const { probabilities, primary_diagnosis, severity, inference_time_seconds } = results
  const sevStyle = SEVERITY_STYLES[severity.level] || SEVERITY_STYLES.Normal

  const chartData = Object.entries(probabilities).map(([name, value]) => ({
    name,
    probability: value,
    fill: COLORS[name] || '#64748b',
  }))

  return (
    <div className="glass-card p-6 animate-slide-up space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-dark-200 uppercase tracking-wider flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-primary-400" />
          Analysis Results
        </h2>
        <div className="flex items-center gap-2 text-xs text-dark-400">
          <Clock className="w-3.5 h-3.5" />
          {inference_time_seconds}s
        </div>
      </div>

      {/* Severity + Diagnosis Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Primary Diagnosis */}
        <div className="glass-card-lighter p-4">
          <p className="text-[11px] text-dark-400 uppercase tracking-wider font-medium mb-2">Primary Finding</p>
          <div className="flex items-center gap-3">
            {primary_diagnosis === 'Normal' ? (
              <CheckCircle className="w-6 h-6 text-emerald-400" />
            ) : (
              <AlertCircle className="w-6 h-6" style={{ color: COLORS[primary_diagnosis] }} />
            )}
            <span className="text-xl font-bold text-dark-100">{primary_diagnosis}</span>
          </div>
        </div>

        {/* Severity */}
        <div className={`glass-card-lighter p-4 border ${sevStyle.border}`}>
          <p className="text-[11px] text-dark-400 uppercase tracking-wider font-medium mb-2">Severity Level</p>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className={`severity-pulse inline-block w-3 h-3 rounded-full`}
                style={{ backgroundColor: severity.color }} />
              <span className={`text-xl font-bold ${sevStyle.text}`}>{severity.level}</span>
            </div>
            <span className="text-xs text-dark-400 font-mono">{(severity.score * 100).toFixed(1)}%</span>
          </div>
          <p className="text-xs text-dark-400 mt-2 leading-relaxed">{severity.description}</p>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="space-y-3">
        <p className="text-[11px] text-dark-400 uppercase tracking-wider font-medium">Class Probabilities</p>
        {Object.entries(probabilities).map(([name, value]) => (
          <div key={name} className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-dark-200 font-medium">{name}</span>
              <span className="font-mono text-dark-300">{(value * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 bg-dark-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: `${Math.max(value * 100, 1)}%`,
                  backgroundColor: COLORS[name],
                  boxShadow: `0 0 8px ${COLORS[name]}40`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} barSize={40}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={{ stroke: '#334155' }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#334155' }} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
