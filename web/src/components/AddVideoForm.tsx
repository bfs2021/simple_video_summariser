import { useRef, useState, type FormEvent } from 'react'
import { Link } from 'react-router'
import { ApiError, type components } from '../api'
import { useSubmitVideo } from '../hooks/useSubmitVideo'
import { parseVideoId } from '../lib/videoId'

type VideoSubmission = components['schemas']['VideoSubmission']

const ERROR_ID = 'add-video-error'

function errorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return 'Could not add the video. Try again.'
  if (error.kind === 'network') return 'Could not add the video. Check your connection and try again.'
  if (error.kind !== 'http') return 'Could not add the video. Try again.'
  if (error.status === 422) return 'url must be a YouTube video ID or video URL'
  if (error.status === 429) return 'Too many requests. Wait a moment and try again.'
  if (error.status === 503) return 'The service is unavailable. Try again shortly.'
  return 'Could not add the video. Try again.'
}

export function AddVideoForm() {
  const [value, setValue] = useState('')
  const [outcome, setOutcome] = useState<VideoSubmission | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const pendingRef = useRef(false)
  const mutation = useSubmitVideo()
  const message = mutation.isError ? errorMessage(mutation.error) : null

  const clearFeedback = () => {
    setOutcome(null)
    mutation.reset()
  }

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (pendingRef.current) return
    pendingRef.current = true
    clearFeedback()
    mutation.mutate(
      { url: value.trim() },
      {
        onSuccess: (submission) => {
          setOutcome(submission)
          setValue('')
        },
        onError: () => inputRef.current?.focus(),
        onSettled: () => {
          pendingRef.current = false
        },
      },
    )
  }

  const videoId = outcome === null ? null : parseVideoId(outcome.video_id)

  return (
    <form className="card" onSubmit={onSubmit}>
      <label className="field">
        YouTube video URL or ID{' '}
        <input
          ref={inputRef}
          className="input"
          type="text"
          value={value}
          aria-invalid={message === null ? undefined : true}
          aria-describedby={message === null ? undefined : ERROR_ID}
          onChange={(event) => {
            setValue(event.target.value)
            clearFeedback()
          }}
        />
      </label>{' '}
      <button className="btn btn--primary" type="submit" disabled={mutation.isPending}>
        {mutation.isPending ? 'Adding…' : 'Add video'}
      </button>
      {message !== null && (
        <p id={ERROR_ID} className="alert alert--danger" role="alert">
          {message}
        </p>
      )}
      <p aria-live="polite">
        {outcome !== null && (
          <>
            <span>{outcome.created ? 'Queued' : 'Already added'}</span>
            {!outcome.created && <> · <span>{outcome.state}</span></>}
            {videoId !== null && <> · <Link to={`/videos/${videoId}`}>{videoId}</Link></>}
          </>
        )}
      </p>
    </form>
  )
}
