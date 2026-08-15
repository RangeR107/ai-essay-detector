import { useState } from 'react'
import { analyzeEssay, type AnalyzeResponse } from './api'
import EssayInput from './components/EssayInput'
import EvidencePanel from './components/EvidencePanel'
import SentenceHighlighter from './components/SentenceHighlighter'
import TokenHeatmap from './components/TokenHeatmap'
import VerdictBadge from './components/VerdictBadge'

function App() {
  const [essayText, setEssayText] = useState('')
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit() {
    setLoading(true)
    setError(null)
    try {
      const res = await analyzeEssay(essayText)
      setResult(res)
      setSelectedIndex(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <header className="mb-10 flex items-start gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 shadow-lg shadow-violet-500/30">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="10.5" cy="10.5" r="6" stroke="white" strokeWidth="2" />
            <line x1="15.2" y1="15.2" x2="20" y2="20" stroke="white" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <h1 className="font-display text-2xl font-bold tracking-tight text-slate-900">
            AI Admissions Essay Detector
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Paste an essay. See exactly which sentences look machine-written, and the evidence behind every flag.
          </p>
        </div>
      </header>

      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-sm shadow-slate-200/50 backdrop-blur-sm">
        <EssayInput value={essayText} onChange={setEssayText} onSubmit={handleSubmit} loading={loading} />
      </div>

      {error && (
        <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-600">{error}</p>
      )}

      {result && (
        <div className="animate-rise-in mt-8 flex flex-col gap-6">
          <VerdictBadge verdict={result.verdict} essayLevelFeatures={result.essay_level_features} sentences={result.sentences} />
          <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-sm shadow-slate-200/50 backdrop-blur-sm">
            <SentenceHighlighter
              sentences={result.sentences}
              transitions={result.transitions}
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
            />
          </div>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-sm shadow-slate-200/50 backdrop-blur-sm">
              <EvidencePanel sentence={selectedIndex !== null ? result.sentences[selectedIndex] : null} />
            </div>
            <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-sm shadow-slate-200/50 backdrop-blur-sm">
              <TokenHeatmap sentence={selectedIndex !== null ? result.sentences[selectedIndex] : null} />
            </div>
          </div>
        </div>
      )}

      <footer className="mt-14 border-t border-slate-200/70 pt-5 text-xs text-slate-400">
        Every verdict traces back to a computed feature value, a percentile against real human writing, or a
        coefficient contribution — never free text from a model. See{' '}
        <code className="rounded bg-slate-100 px-1 py-0.5 text-slate-500">docs/LIMITATIONS.md</code> for what's still
        unproven.
      </footer>
    </div>
  )
}

export default App
