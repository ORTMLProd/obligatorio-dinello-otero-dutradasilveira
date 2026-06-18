import { describe, expect, it } from 'vitest'

import { clampIndex, confidencePct, sortedByValueDesc } from './format'

describe('sortedByValueDesc', () => {
  it('orders entries by value descending', () => {
    expect(sortedByValueDesc({ a: 0.1, b: 0.7, c: 0.2 })).toEqual([
      ['b', 0.7],
      ['c', 0.2],
      ['a', 0.1],
    ])
  })
})

describe('confidencePct', () => {
  it('rounds the label probability to a percentage', () => {
    expect(confidencePct({ corner: 0.713, goal: 0.287 }, 'corner')).toBe(71)
  })
  it('returns 0 for a missing label', () => {
    expect(confidencePct({ corner: 0.7 }, 'goal')).toBe(0)
  })
})

describe('clampIndex', () => {
  it('clamps to [0, length-1] and handles empty', () => {
    expect(clampIndex(5, 3)).toBe(2)
    expect(clampIndex(-1, 3)).toBe(0)
    expect(clampIndex(1, 3)).toBe(1)
    expect(clampIndex(2, 0)).toBe(0)
  })
})
