import React, { useState, useRef, useMemo } from 'react'
import type { ChartDataPoint } from '../types'
import { Activity } from 'lucide-react'

interface InteractiveChartsProps {
  data: ChartDataPoint[]
  targetNote?: string
}

export const InteractiveCharts: React.FC<InteractiveChartsProps> = ({ data, targetNote = 'D#4' }) => {
  const [activeTab, setActiveTab] = useState<'both' | 'pitch' | 'mar'>('both')
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Filter atau sampling jika data terlalu padat
  const points = useMemo(() => data, [data])

  const minT = points && points.length > 0 ? points[0].t : 0
  const maxT = points && points.length > 0 ? points[points.length - 1].t || 1 : 1
  const duration = maxT - minT || 1

  // Hitung range cent (biasanya -100 s/d +100 atau min/max cent)
  const centValues = (points || [])
    .filter((p) => p.cent !== null && p.stable)
    .map((p) => p.cent as number)

  const minCent = centValues.length ? Math.min(-50, Math.floor(Math.min(...centValues) / 25) * 25) : -50
  const maxCent = centValues.length ? Math.max(50, Math.ceil(Math.max(...centValues) / 25) * 25) : 50
  const centRange = maxCent - minCent || 100

  // Hitung range MAR (biasanya 0.0 s/d 1.0)
  const marValues = (points || [])
    .filter((p) => p.mar_smooth !== null)
    .map((p) => p.mar_smooth as number)

  const minMar = 0
  const maxMar = marValues.length ? Math.max(0.6, Math.ceil(Math.max(...marValues) * 10) / 10 + 0.1) : 0.8
  const marRange = maxMar - minMar || 1

  // Dimensi SVG
  const width = 1000
  const heightPitch = 240
  const heightMar = 160
  const padL = 60
  const padR = 20
  const padT = 20
  const padB = 30

  const plotW = width - padL - padR
  const plotHPitch = heightPitch - padT - padB
  const plotHMar = heightMar - padT - padB

  // Fungsi konversi koordinat
  const getX = (t: number) => padL + ((t - minT) / duration) * plotW
  const getYCent = (cent: number) => padT + plotHPitch - ((cent - minCent) / centRange) * plotHPitch
  const getYMar = (mar: number) => padT + plotHMar - ((mar - minMar) / marRange) * plotHMar

  // Jalur Pitch
  const pitchSegments = useMemo(() => {
    if (!points || points.length === 0) return []
    const segments: Array<{ points: string; color: string; status: string }> = []
    let currentSegment: { points: string[]; color: string; status: string } | null = null

    points.forEach((p) => {
      if (p.cent !== null && p.stable) {
        const x = getX(p.t).toFixed(1)
        const y = getYCent(p.cent).toFixed(1)
        const pointStr = `${x},${y}`

        let color = '#10b981' // in-tune (emerald)
        if (p.status === 'flat') color = '#f59e0b' // amber
        else if (p.status === 'sharp') color = '#f43f5e' // rose
        else if (p.status === 'nada-salah') color = '#ef4444' // red

        if (!currentSegment || currentSegment.status !== p.status) {
          if (currentSegment && currentSegment.points.length > 1) {
            segments.push({
              points: currentSegment.points.join(' '),
              color: currentSegment.color,
              status: currentSegment.status,
            })
          }
          currentSegment = { points: [pointStr], color, status: p.status }
        } else {
          currentSegment.points.push(pointStr)
        }
      } else {
        if (currentSegment && currentSegment.points.length > 1) {
          segments.push({
            points: currentSegment.points.join(' '),
            color: currentSegment.color,
            status: currentSegment.status,
          })
        }
        currentSegment = null
      }
    })

    if (currentSegment && (currentSegment as { points: string[]; color: string; status: string }).points.length > 1) {
      const seg = currentSegment as { points: string[]; color: string; status: string }
      segments.push({
        points: seg.points.join(' '),
        color: seg.color,
        status: seg.status,
      })
    }

    return segments
  }, [points, minT, duration, minCent, centRange])

  // Jalur MAR Smooth & Interp
  const marSmoothPath = useMemo(() => {
    if (!points || points.length === 0) return ''
    const pts = points
      .filter((p) => p.mar_smooth !== null)
      .map((p) => `${getX(p.t).toFixed(1)},${getYMar(p.mar_smooth as number).toFixed(1)}`)
    return pts.length > 1 ? `M ${pts.join(' L ')}` : ''
  }, [points, minT, duration, minMar, maxMar])

  // Area under curve MAR
  const marAreaPath = useMemo(() => {
    if (!points || points.length === 0) return ''
    const valid = points.filter((p) => p.mar_smooth !== null)
    if (valid.length < 2) return ''
    const pts = valid.map((p) => `${getX(p.t).toFixed(1)},${getYMar(p.mar_smooth as number).toFixed(1)}`)
    const firstX = getX(valid[0].t).toFixed(1)
    const lastX = getX(valid[valid.length - 1].t).toFixed(1)
    const bottomY = (padT + plotHMar).toFixed(1)
    return `M ${firstX},${bottomY} L ${pts.join(' L ')} L ${lastX},${bottomY} Z`
  }, [points, minT, duration, minMar, maxMar])

  if (!points || points.length === 0) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-8 text-center text-slate-400">
        Belum ada data visualisasi timeline.
      </div>
    )
  }

  // Mouse move handler
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const svgX = (mouseX / rect.width) * width
    if (svgX < padL || svgX > padL + plotW) {
      setHoverIndex(null)
      return
    }
    const ratio = (svgX - padL) / plotW
    const hoverT = minT + ratio * duration

    let bestIdx = 0
    let bestDiff = Infinity
    points.forEach((p, idx) => {
      const diff = Math.abs(p.t - hoverT)
      if (diff < bestDiff) {
        bestDiff = diff
        bestIdx = idx
      }
    })
    setHoverIndex(bestIdx)
  }

  const hoverPoint = hoverIndex !== null ? points[hoverIndex] : null

  // In-tune band boundaries
  const yInTuneTop = getYCent(25)
  const yInTuneBottom = getYCent(-25)

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 sm:p-8 backdrop-blur-xl shadow-xl space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center space-x-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            <span>Grafik Sinkronisasi Audio & Visual</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Perbandingan timeline deviasi nada (cent) dengan dinamika bukaan mulut (MAR)
          </p>
        </div>

        <div className="flex items-center space-x-2 bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
          <button
            onClick={() => setActiveTab('both')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'both' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            Semua (Sinkron)
          </button>
          <button
            onClick={() => setActiveTab('pitch')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'pitch' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            Deviasi Pitch
          </button>
          <button
            onClick={() => setActiveTab('mar')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'mar' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            Bukaan Mulut (MAR)
          </button>
        </div>
      </div>

      {/* SVG Timeline Canvas */}
      <div ref={containerRef} className="relative select-none">
        {/* Tooltip Float */}
        {hoverPoint && (
          <div
            className="absolute top-2 right-4 z-20 bg-slate-950/90 border border-slate-700 p-3 rounded-xl shadow-xl backdrop-blur-md pointer-events-none text-xs flex items-center space-x-4"
          >
            <div>
              <span className="text-slate-400 block font-mono">Waktu:</span>
              <span className="font-bold text-white">{hoverPoint.t.toFixed(2)} detik</span>
            </div>
            {hoverPoint.cent !== null && (
              <div>
                <span className="text-slate-400 block font-mono">Deviasi:</span>
                <span
                  className={`font-bold font-mono ${
                    Math.abs(hoverPoint.cent) <= 25
                      ? 'text-emerald-400'
                      : hoverPoint.cent < 0
                      ? 'text-amber-400'
                      : 'text-rose-400'
                  }`}
                >
                  {hoverPoint.cent > 0 ? `+${hoverPoint.cent}` : hoverPoint.cent} cent
                </span>
              </div>
            )}
            {hoverPoint.f0 && (
              <div>
                <span className="text-slate-400 block font-mono">Pitch f0:</span>
                <span className="font-bold text-indigo-300 font-mono">{hoverPoint.f0} Hz</span>
              </div>
            )}
            {hoverPoint.mar_smooth !== null && (
              <div>
                <span className="text-slate-400 block font-mono">MAR:</span>
                <span className="font-bold text-violet-400 font-mono">{hoverPoint.mar_smooth.toFixed(3)}</span>
              </div>
            )}
            <div>
              <span className="text-slate-400 block font-mono">Status:</span>
              <span className="font-semibold uppercase tracking-wider text-[11px] text-white">
                {hoverPoint.status}
              </span>
            </div>
          </div>
        )}

        <div className="space-y-4">
          {/* Chart 1: Pitch Deviation */}
          {(activeTab === 'both' || activeTab === 'pitch') && (
            <div className="bg-slate-950/60 rounded-2xl border border-slate-800 p-4 relative overflow-hidden">
              <div className="flex items-center justify-between mb-2 px-2">
                <span className="text-xs font-semibold text-slate-300 flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block" />
                  <span>Deviasi Nada Terhadap Target ({targetNote})</span>
                </span>
                <div className="flex items-center space-x-3 text-[11px] text-slate-400">
                  <span className="flex items-center space-x-1">
                    <span className="w-2 h-2 rounded bg-emerald-500/30 border border-emerald-500" />
                    <span>Zona In-Tune (±25 cent)</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <span className="w-2 h-0.5 bg-slate-500" />
                    <span>Acuan (0 cent)</span>
                  </span>
                </div>
              </div>

              <svg
                viewBox={`0 0 ${width} ${heightPitch}`}
                className="w-full h-auto cursor-crosshair overflow-visible"
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoverIndex(null)}
              >
                {/* Background in-tune zone */}
                <rect
                  x={padL}
                  y={yInTuneTop}
                  width={plotW}
                  height={Math.max(0, yInTuneBottom - yInTuneTop)}
                  fill="rgba(16, 185, 129, 0.08)"
                  stroke="rgba(16, 185, 129, 0.25)"
                  strokeDasharray="4 4"
                />

                {/* Grid horizontal lines */}
                {[maxCent, 50, 25, 0, -25, -50, minCent].map((val) => {
                  const y = getYCent(val)
                  const isZero = val === 0
                  return (
                    <g key={val}>
                      <line
                        x1={padL}
                        y1={y}
                        x2={padL + plotW}
                        y2={y}
                        stroke={isZero ? 'rgba(255, 255, 255, 0.3)' : 'rgba(255, 255, 255, 0.07)'}
                        strokeWidth={isZero ? 1.5 : 1}
                      />
                      <text
                        x={padL - 8}
                        y={y + 4}
                        fill="rgba(148, 163, 184, 0.7)"
                        fontSize="10"
                        fontFamily="monospace"
                        textAnchor="end"
                      >
                        {val > 0 ? `+${val}` : val}
                      </text>
                    </g>
                  )
                })}

                {/* X axis ticks */}
                {Array.from({ length: 6 }).map((_, i) => {
                  const t = minT + (i / 5) * duration
                  const x = getX(t)
                  return (
                    <g key={i}>
                      <line x1={x} y1={padT + plotHPitch} x2={x} y2={padT + plotHPitch + 5} stroke="rgba(255,255,255,0.2)" />
                      <text
                        x={x}
                        y={padT + plotHPitch + 18}
                        fill="rgba(148, 163, 184, 0.7)"
                        fontSize="10"
                        fontFamily="monospace"
                        textAnchor="middle"
                      >
                        {t.toFixed(1)}s
                      </text>
                    </g>
                  )
                })}

                {/* Pitch Lines */}
                {pitchSegments.map((seg, i) => (
                  <polyline
                    key={i}
                    points={seg.points}
                    fill="none"
                    stroke={seg.color}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                ))}

                {/* Crosshair indicator */}
                {hoverPoint && (
                  <line
                    x1={getX(hoverPoint.t)}
                    y1={padT}
                    x2={getX(hoverPoint.t)}
                    y2={padT + plotHPitch}
                    stroke="rgba(255, 255, 255, 0.6)"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                )}
                {hoverPoint && hoverPoint.cent !== null && (
                  <circle
                    cx={getX(hoverPoint.t)}
                    cy={getYCent(hoverPoint.cent)}
                    r="5"
                    fill="#ffffff"
                    stroke={
                      Math.abs(hoverPoint.cent) <= 25
                        ? '#10b981'
                        : hoverPoint.cent < 0
                        ? '#f59e0b'
                        : '#f43f5e'
                    }
                    strokeWidth="2"
                  />
                )}
              </svg>
            </div>
          )}

          {/* Chart 2: Mouth Aspect Ratio (MAR) */}
          {(activeTab === 'both' || activeTab === 'mar') && (
            <div className="bg-slate-950/60 rounded-2xl border border-slate-800 p-4 relative overflow-hidden">
              <div className="flex items-center justify-between mb-2 px-2">
                <span className="text-xs font-semibold text-slate-300 flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-violet-400 inline-block" />
                  <span>Dinamika Bukaan Mulut (Mouth Aspect Ratio — MAR)</span>
                </span>
                <span className="text-[11px] text-slate-400">
                  MAR = Vertikal Bibir Dalam / Horizontal Sudut Bibir
                </span>
              </div>

              <svg
                viewBox={`0 0 ${width} ${heightMar}`}
                className="w-full h-auto cursor-crosshair overflow-visible"
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoverIndex(null)}
              >
                {/* Horizontal Grid lines */}
                {[0, 0.2, 0.4, 0.6, 0.8].map((val) => {
                  if (val > maxMar) return null
                  const y = getYMar(val)
                  return (
                    <g key={val}>
                      <line
                        x1={padL}
                        y1={y}
                        x2={padL + plotW}
                        y2={y}
                        stroke="rgba(255, 255, 255, 0.07)"
                        strokeWidth="1"
                      />
                      <text
                        x={padL - 8}
                        y={y + 3}
                        fill="rgba(148, 163, 184, 0.7)"
                        fontSize="10"
                        fontFamily="monospace"
                        textAnchor="end"
                      >
                        {val.toFixed(1)}
                      </text>
                    </g>
                  )
                })}

                {/* X axis ticks */}
                {Array.from({ length: 6 }).map((_, i) => {
                  const t = minT + (i / 5) * duration
                  const x = getX(t)
                  return (
                    <g key={i}>
                      <line x1={x} y1={padT + plotHMar} x2={x} y2={padT + plotHMar + 5} stroke="rgba(255,255,255,0.2)" />
                      <text
                        x={x}
                        y={padT + plotHMar + 16}
                        fill="rgba(148, 163, 184, 0.7)"
                        fontSize="10"
                        fontFamily="monospace"
                        textAnchor="middle"
                      >
                        {t.toFixed(1)}s
                      </text>
                    </g>
                  )
                })}

                {/* MAR Area */}
                {marAreaPath && (
                  <path d={marAreaPath} fill="rgba(139, 92, 246, 0.15)" />
                )}

                {/* MAR Line */}
                {marSmoothPath && (
                  <path
                    d={marSmoothPath}
                    fill="none"
                    stroke="#a855f7"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )}

                {/* Crosshair indicator */}
                {hoverPoint && (
                  <line
                    x1={getX(hoverPoint.t)}
                    y1={padT}
                    x2={getX(hoverPoint.t)}
                    y2={padT + plotHMar}
                    stroke="rgba(255, 255, 255, 0.6)"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                )}
                {hoverPoint && hoverPoint.mar_smooth !== null && (
                  <circle
                    cx={getX(hoverPoint.t)}
                    cy={getYMar(hoverPoint.mar_smooth)}
                    r="5"
                    fill="#ffffff"
                    stroke="#a855f7"
                    strokeWidth="2"
                  />
                )}
              </svg>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
