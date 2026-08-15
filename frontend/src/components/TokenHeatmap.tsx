import type { SentenceResult } from '../api'

interface Props {
  sentence: SentenceResult | null
}

// Darker = the model was less confident this token was coming (higher GLTR
// rank bucket). This is the raw GPT-2 signal, shown as-is — it's an
// instrument reading, not a verdict (plan §0).
const BUCKET_STYLES: Record<string, string> = {
  top10: 'bg-emerald-100 text-emerald-900',
  top100: 'bg-lime-100 text-lime-900',
  top1000: 'bg-amber-200 text-amber-900',
  rest: 'bg-rose-300 text-rose-950',
}

const BUCKET_LABELS: Record<string, string> = {
  top10: 'top 10',
  top100: 'top 100',
  top1000: 'top 1,000',
  rest: 'beyond top 1,000',
}

function PanelHeading() {
  return (
    <div className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="10" width="4" height="8" rx="1" fill="currentColor" />
        <rect x="10" y="5" width="4" height="13" rx="1" fill="currentColor" />
        <rect x="17" y="13" width="4" height="5" rx="1" fill="currentColor" />
      </svg>
      Token predictability
    </div>
  )
}

export default function TokenHeatmap({ sentence }: Props) {
  if (!sentence) {
    return (
      <div>
        <PanelHeading />
        <p className="text-sm text-slate-400">Click a sentence above to see its token-level GLTR heatmap.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <div>
        <PanelHeading />
        <p className="text-xs text-slate-500">
          Each word shaded by GPT-2's predicted rank for that token — a raw model reading, not a verdict.
        </p>
      </div>
      <p className="whitespace-pre-wrap rounded-lg bg-slate-50 p-3 font-mono text-sm leading-loose">
        {sentence.token_ranks.map((t, i) => (
          <span
            key={i}
            title={`rank bucket: ${BUCKET_LABELS[t.rank_bucket] ?? t.rank_bucket}`}
            className={`rounded px-0.5 transition-transform hover:scale-105 ${BUCKET_STYLES[t.rank_bucket] ?? ''}`}
          >
            {t.token}
          </span>
        ))}
      </p>
      <div className="flex flex-wrap gap-3 text-xs text-slate-500">
        {Object.entries(BUCKET_LABELS).map(([bucket, label]) => (
          <span key={bucket} className="flex items-center gap-1">
            <span className={`inline-block h-3 w-3 rounded ${BUCKET_STYLES[bucket]}`} />
            {label}
          </span>
        ))}
      </div>
    </div>
  )
}
