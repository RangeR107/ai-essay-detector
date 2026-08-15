import type { SentenceResult } from '../api'

interface Props {
  sentence: SentenceResult | null
}

function phrase(percentile: number): string {
  if (percentile >= 50) {
    return `higher than ${percentile.toFixed(0)}% of human training sentences`
  }
  return `lower than ${(100 - percentile).toFixed(0)}% of human training sentences`
}

const DIRECTION_STYLES: Record<string, string> = {
  'ai-like': 'bg-rose-400',
  'human-like': 'bg-emerald-400',
}

function PanelHeading({ children }: { children: string }) {
  return (
    <div className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
        <path d="M12 3v3m0 12v3m9-9h-3M6 12H3m14.5-6.5l-2 2m-9 9l-2 2m0-13l2 2m9 9l2 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
      Evidence
    </div>
  )
}

export default function EvidencePanel({ sentence }: Props) {
  if (!sentence) {
    return (
      <div>
        <PanelHeading>Evidence</PanelHeading>
        <p className="text-sm text-slate-400">Click a sentence above to see why it scored the way it did.</p>
      </div>
    )
  }

  if (sentence.top_features.length === 0) {
    return (
      <div>
        <PanelHeading>Evidence</PanelHeading>
        <p className="text-sm text-slate-400">No evidence available for this sentence.</p>
      </div>
    )
  }

  const maxMagnitude = Math.max(...sentence.top_features.map((f) => f.magnitude))

  return (
    <div className="flex flex-col gap-3">
      <div>
        <PanelHeading>Evidence</PanelHeading>
        <p className="text-xs text-slate-500">
          Top contributing features, computed by measuring how much each one moves this sentence's score — not a
          generated explanation.
        </p>
      </div>
      <ul className="flex flex-col gap-3.5">
        {sentence.top_features.map((f, i) => (
          <li key={i} className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-slate-800">{f.name}</span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                  f.direction === 'ai-like' ? 'bg-rose-50 text-rose-600' : 'bg-emerald-50 text-emerald-600'
                }`}
              >
                {f.direction}
              </span>
            </div>
            <p className="text-xs text-slate-500">{phrase(f.percentile)}</p>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-1.5 rounded-full transition-all duration-500 ${DIRECTION_STYLES[f.direction]}`}
                style={{ width: `${maxMagnitude > 0 ? (f.magnitude / maxMagnitude) * 100 : 0}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
