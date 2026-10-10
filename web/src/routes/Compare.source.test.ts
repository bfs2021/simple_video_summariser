import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

// Mechanical guards for #53 over the files it adds: nothing stored or taken
// from the URL may become markup, and every request goes through the typed request helper (#118).
const files = [
  'routes/Compare.tsx',
  'hooks/useAnalyses.ts',
  'lib/compare.ts',
  'components/ClaimItem.tsx',
].map((rel) => ({
  rel,
  text: readFileSync(fileURLToPath(new URL(`../${rel}`, import.meta.url)), 'utf8'),
}))

describe('Compare source scan', () => {
  it.each(files)('$rel never injects HTML', ({ text }) => {
    expect(text).not.toMatch(/dangerouslySetInnerHTML|innerHTML|insertAdjacentHTML|outerHTML/)
  })

  it('every request goes through the typed helper, which builds the URL with apiUrl()', () => {
    const hook = files.find((f) => f.rel === 'hooks/useAnalyses.ts')?.text ?? ''
    expect(hook).toMatch(/api\s*\.get\(/)
    expect(hook).not.toMatch(/\bfetch\(/)
    expect(hook).not.toMatch(/['"`]\/api\//)
  })

  it('the view has no seek or timestamp code of its own', () => {
    for (const { text } of files) {
      expect(text).not.toMatch(/\bseekTo\s*\(/)
      expect(text).not.toMatch(/function formatTimestamp/)
    }
  })

  it('does not hand-memoise: the React Compiler does that', () => {
    for (const { text } of files) expect(text).not.toMatch(/\buse(Memo|Callback)\(/)
  })
})
