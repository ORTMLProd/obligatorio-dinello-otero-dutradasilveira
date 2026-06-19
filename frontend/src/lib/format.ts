// Pure UI helpers, unit-tested with Vitest. Kept out of components so they can be
// reasoned about and tested in isolation.

/** Entries of a probability map, sorted by value descending. */
export function sortedByValueDesc(probabilities: Record<string, number>): [string, number][] {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1])
}

/** The label's probability as a rounded percentage (0 if the label is absent). */
export function confidencePct(probabilities: Record<string, number>, label: string): number {
  return Math.round((probabilities[label] ?? 0) * 100)
}

/** Clamp ``index`` to ``[0, length - 1]`` (0 when ``length`` is 0). */
export function clampIndex(index: number, length: number): number {
  if (length <= 0) return 0
  return Math.min(Math.max(index, 0), length - 1)
}
