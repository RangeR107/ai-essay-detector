import type { SentenceResult, Transition } from '../api'

interface Props {
  sentences: SentenceResult[]
  transitions: Transition[]
  selectedIndex: number | null
  onSelect: (index: number) => void
}

// Continuous scale, not a binary red/green split (plan §7): 0 -> human-like,
// 1 -> ai-like, interpolated through an amber midpoint.
function scoreToColor(score: number): string {
  const clamped = Math.max(0, Math.min(1, score))
  const human = [167, 243, 208] // emerald-200
  const ai = [254, 205, 211] // rose-200
  const mid = [253, 230, 138] // amber-200
  const [from, to, t] =
    clamped < 0.5 ? [human, mid, clamped * 2] : [mid, ai, (clamped - 0.5) * 2]
  const rgb = from.map((c, i) => Math.round(c + (to[i] - c) * t))
  return `rgb(${rgb.join(',')})`
}

export default function SentenceHighlighter({ sentences, transitions, selectedIndex, onSelect }: Props) {
  const transitionByIndex = new Map(transitions.map((t) => [t.sentence_index, t.note]))

  return (
    <p className="whitespace-pre-wrap leading-relaxed text-sm">
      {sentences.map((s, i) => (
        <span key={i} className="relative">
          <span
            onClick={() => onSelect(i)}
            title={
              s.context_merged
                ? `AI score: ${(s.ai_score * 100).toFixed(0)}% — too short to score alone, reflects a merged span with a neighboring sentence`
                : `AI score: ${(s.ai_score * 100).toFixed(0)}%`
            }
            style={{ backgroundColor: scoreToColor(s.ai_score) }}
            className={`cursor-pointer rounded px-0.5 decoration-dotted transition-all hover:brightness-95 ${
              selectedIndex === i ? 'ring-2 ring-violet-500 ring-offset-1' : ''
            } ${s.context_merged ? 'underline decoration-slate-400' : ''}`}
          >
            {s.text}
          </span>
          {transitionByIndex.has(i) && (
            <span
              className="ml-1 cursor-help text-amber-600"
              title={transitionByIndex.get(i)}
            >
              ⚑
            </span>
          )}
          {' '}
        </span>
      ))}
    </p>
  )
}
