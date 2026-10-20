import { expect, inject, it } from 'vitest'

declare module 'vitest' {
  export interface ProvidedContext {
    effectiveFileIsolation: boolean
  }
}

it('runs every test file in isolation', () => {
  expect(
    inject('effectiveFileIsolation'),
    'file isolation must remain enabled so state cannot leak between test files',
  ).toBe(true)
})
