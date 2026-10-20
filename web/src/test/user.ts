import userEvent from '@testing-library/user-event'
import { act } from '@testing-library/react'
import { vi } from 'vitest'

export function setupUser() {
  const user = userEvent.setup(
    vi.isFakeTimers() ? { advanceTimers: vi.advanceTimersByTime } : undefined,
  )
  async function run<T>(action: () => Promise<T>): Promise<T> {
    if (!vi.isFakeTimers()) return action()
    let result!: T
    await act(async () => {
      const pending = action()
      await vi.advanceTimersByTimeAsync(0)
      result = await pending
    })
    return result
  }
  return {
    click: (...args: Parameters<typeof user.click>) => run(() => user.click(...args)),
    clear: (...args: Parameters<typeof user.clear>) => run(() => user.clear(...args)),
    keyboard: (...args: Parameters<typeof user.keyboard>) => run(() => user.keyboard(...args)),
    selectOptions: (...args: Parameters<typeof user.selectOptions>) => run(() => user.selectOptions(...args)),
    tab: (...args: Parameters<typeof user.tab>) => run(() => user.tab(...args)),
    type: (...args: Parameters<typeof user.type>) => run(() => user.type(...args)),
  }
}
