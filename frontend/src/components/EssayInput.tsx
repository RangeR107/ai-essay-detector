interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  loading: boolean
}

export default function EssayInput({ value, onChange, onSubmit, loading }: Props) {
  return (
    <div className="flex flex-col gap-3">
      <textarea
        className="min-h-64 w-full resize-y rounded-xl border border-slate-200 bg-white p-4 text-sm leading-relaxed text-slate-800 placeholder:text-slate-400 transition-shadow focus:border-violet-400 focus:outline-none focus:ring-4 focus:ring-violet-100"
        placeholder="Paste an admissions essay to analyze..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      <div className="flex items-center gap-3">
        <button
          onClick={onSubmit}
          disabled={loading || value.trim().length === 0}
          className="group inline-flex items-center gap-2 self-start rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-violet-500/25 transition-all hover:shadow-lg hover:shadow-violet-500/35 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none disabled:active:scale-100"
        >
          {loading ? (
            <>
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />
              Analyzing…
            </>
          ) : (
            <>
              Analyze
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                className="transition-transform group-hover:translate-x-0.5"
              >
                <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </>
          )}
        </button>
        {value.trim().length > 0 && !loading && (
          <span className="text-xs text-slate-400">{value.trim().split(/\s+/).length} words</span>
        )}
      </div>
    </div>
  )
}
