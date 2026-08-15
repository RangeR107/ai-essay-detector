export interface FeatureContribution {
  name: string
  percentile: number
  direction: 'ai-like' | 'human-like'
  // Only meaningful relative to the other entries in the same sentence's
  // top_features array, not comparable across sentences.
  magnitude: number
}

export interface TokenRank {
  token: string
  rank_bucket: 'top10' | 'top100' | 'top1000' | 'rest'
}

export interface SentenceResult {
  text: string
  start: number
  end: number
  ai_score: number
  top_features: FeatureContribution[]
  token_ranks: TokenRank[]
  // Too short (< 4 words) to score alone — this sentence's score/evidence
  // reflects a merged span with a neighboring sentence instead. Still a
  // real, context-informed score, not an excluded/discounted one.
  context_merged: boolean
}

export interface Transition {
  sentence_index: number
  note: string
}

export interface VerdictResult {
  label: 'Likely Human' | 'Inconclusive' | 'Likely AI'
  confidence: number
}

export interface EssayLevelFeatures {
  burstiness: number
  sentence_length_variance: number
  score_volatility: number
}

export interface AnalyzeResponse {
  sentences: SentenceResult[]
  transitions: Transition[]
  verdict: VerdictResult
  essay_level_features: EssayLevelFeatures
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export async function analyzeEssay(essayText: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ essay_text: essayText }),
  })
  if (!res.ok) {
    throw new Error(`Analyze request failed: ${res.status}`)
  }
  return res.json()
}
