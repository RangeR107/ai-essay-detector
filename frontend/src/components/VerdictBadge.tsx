import type { ReactNode } from 'react'
import type { EssayLevelFeatures, SentenceResult, VerdictResult } from '../api'

interface Props {
  verdict: VerdictResult
  essayLevelFeatures: EssayLevelFeatures
  sentences: SentenceResult[]
}

const VERDICT_STYLES: Record<string, { badge: string; glow: string; icon: ReactNode; bar: string }> = {
  'Likely Human': {
    badge: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    glow: 'shadow-emerald-500/20',
    bar: 'bg-emerald-400',
    icon: (
      <path d="M5 13l4 4L19 7" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    ),
  },
  Inconclusive: {
    badge: 'bg-amber-50 text-amber-800 border-amber-200',
    glow: 'shadow-amber-500/20',
    bar: 'bg-amber-400',
    icon: (
      <>
        <path d="M12 8v5" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
        <circle cx="12" cy="16.2" r="1.1" fill="currentColor" />
      </>
    ),
  },
  'Likely AI': {
    badge: 'bg-rose-50 text-rose-800 border-rose-200',
    glow: 'shadow-rose-500/20',
    bar: 'bg-rose-400',
    icon: (
      <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
    ),
  },
}

// A stat with a plain-language explainer, shown on hover. Deliberately a
// real CSS tooltip (group-hover, opacity/visibility toggle) rather than
// the native `title` attribute — title tooltips have a ~1s delay, render
// inconsistently across browsers, and don't work on touch at all, which
// makes them a bad fit for something meant to actually be read.
function Stat({ label, value, explainer }: { label: string; value: string; explainer: string }) {
  return (
    <span className="group relative flex cursor-help items-center gap-1.5">
      <span className="font-medium text-slate-700">{value}</span>
      {label}
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 w-56 -translate-x-1/2 rounded-lg bg-slate-800 px-3 py-2 text-left text-xs leading-relaxed font-normal text-white opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100"
      >
        {explainer}
        <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
      </span>
    </span>
  )
}

export default function VerdictBadge({ verdict, essayLevelFeatures, sentences }: Props) {
  const style = VERDICT_STYLES[verdict.label]
  const meanScore =
    sentences.length > 0 ? sentences.reduce((sum, s) => sum + s.ai_score, 0) / sentences.length : 0

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-4">
        <span
          className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold shadow-sm ${style?.badge ?? ''} ${style?.glow ?? ''}`}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            {style?.icon}
          </svg>
          {verdict.label}
        </span>
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
          <Stat
            label="confidence"
            value={`${(verdict.confidence * 100).toFixed(0)}%`}
            explainer="How far the essay's average sentence score sits from the undecided midpoint — not how sure the tool is that its label is correct. Low confidence means the average score landed close to the Inconclusive band's center, i.e. genuinely mixed signal."
          />
          <span className="h-3 w-px bg-slate-200" />
          <Stat
            label="burstiness"
            value={essayLevelFeatures.burstiness.toFixed(1)}
            explainer="How much sentence-to-sentence perplexity varies across the essay. Human writing tends to be 'bursty' — some sentences much more predictable than others; AI writing tends to be more evenly predictable throughout."
          />
          <span className="h-3 w-px bg-slate-200" />
          <Stat
            label="volatility"
            value={essayLevelFeatures.score_volatility.toFixed(2)}
            explainer="How much the AI-likelihood score itself swings between sentences. Only a mild, tail-only signal (see docs/LIMITATIONS.md) — a high value slightly dampens confidence rather than changing the verdict."
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-2 rounded-full transition-all duration-700 ${style?.bar ?? 'bg-slate-400'}`}
            style={{ width: `${Math.round(meanScore * 100)}%` }}
          />
        </div>
        <span
          className="w-28 shrink-0 text-right text-xs text-slate-500"
          title="Mean AI-likelihood across every sentence — the raw number behind the verdict above, shown for transparency, not as a substitute for it. A single percentage on its own can't be argued with; the sentence-by-sentence evidence below is where the actual case is made."
        >
          <span className="font-medium text-slate-700">{Math.round(meanScore * 100)}%</span> mean AI-likelihood
        </span>
      </div>
    </div>
  )
}
