import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  Copy,
  FileSearch,
  Files,
  FlaskConical,
  Menu,
  MessageSquareText,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react'

import { apiRequest } from './api/client'
import { askResearchQuestion } from './api/researchApi'
import {
  getDocuments,
  uploadDocument,
  searchDocuments,
  askDocuments,
} from './api/documentsApi'

const navigation = [
  { label: 'Research Assistant', icon: BrainCircuit, id: 'research' },
  { label: 'PDF Library', icon: Files, id: 'documents' },
  { label: 'Document Search', icon: FileSearch, id: 'search' },
  {
    label: 'PDF Question Answering',
    icon: MessageSquareText,
    id: 'pdf-qa',
  },
]

const examples = [
  {
    title: 'Cardiovascular research',
    question:
      'What evidence supports the effectiveness of aspirin in preventing cardiovascular events?',
    icon: Activity,
  },
  {
    title: 'Explore scientific evidence',
    question:
      'What are the major risk factors associated with cardiovascular disease?',
    icon: FlaskConical,
  },
  {
    title: 'Understand research findings',
    question:
      'How can the effectiveness of a scientific intervention be evaluated?',
    icon: BookOpen,
  },
]

function getErrorMessage(error) {
  const message = error?.message || 'Something went wrong.'

  if (/failed to fetch|networkerror/i.test(message)) {
    return 'Cannot connect to the backend. Check that FastAPI is running at http://127.0.0.1:8000 and that CORS allows the frontend origin.'
  }

  if (/429|quota|resource_exhausted/i.test(message)) {
    return 'The AI generation quota may be exhausted. Please try again later. Document search may still work independently.'
  }

  return message
}

function AppButton({
  children,
  onClick,
  disabled = false,
  variant = 'primary',
  type = 'button',
}) {
  const styles =
    variant === 'secondary'
      ? 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
      : 'bg-indigo-600 text-white hover:bg-indigo-700'

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}
    >
      {children}
    </button>
  )
}

function Sidebar({ page, setPage, open, close }) {
  const systemNavigation = [
    { label: 'API Status', icon: ShieldCheck, id: 'status' },
    { label: 'Settings', icon: Settings2, id: 'settings' },
  ]

  function navigate(id) {
    setPage(id)
    close()
  }

  return (
    <>
      {open && (
        <button
          aria-label="Close navigation overlay"
          onClick={close}
          className="fixed inset-0 z-40 bg-slate-950/40 lg:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[264px] flex-col border-r border-slate-200 bg-white transition-transform duration-200 lg:static lg:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-[76px] items-center justify-between border-b border-slate-100 px-5">
          <button
            onClick={() => navigate('research')}
            className="flex items-center gap-3 text-left"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-200">
              <BrainCircuit size={23} />
            </div>

            <div>
              <p className="font-bold tracking-tight text-slate-900">
                SciFact
              </p>
              <p className="text-xs text-slate-500">
                Research Assistant
              </p>
            </div>
          </button>

          <button
            aria-label="Close sidebar"
            onClick={close}
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 lg:hidden"
          >
            <X size={19} />
          </button>
        </div>

        <div className="px-4 pt-7">
          <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.17em] text-slate-400">
            Workspace
          </p>

          <nav className="space-y-1">
            {navigation.map(({ label, icon: Icon, id }) => (
              <button
                key={id}
                onClick={() => navigate(id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-[13px] font-medium transition ${
                  page === id
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
                }`}
              >
                <Icon size={18} />
                <span>{label}</span>

                {page === id && (
                  <span className="ml-auto h-1.5 w-1.5 rounded-full bg-indigo-600" />
                )}
              </button>
            ))}
          </nav>
        </div>

        <div className="px-4 pt-8">
          <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.17em] text-slate-400">
            System
          </p>

          {systemNavigation.map(({ label, icon: Icon, id }) => (
            <button
              key={id}
              onClick={() => navigate(id)}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-[13px] font-medium ${
                page === id
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </div>

        <div className="mt-auto p-4">
          <div className="rounded-2xl border border-indigo-100 bg-gradient-to-br from-indigo-50 to-white p-4">
            <Sparkles size={22} className="mb-3 text-indigo-600" />

            <p className="text-sm font-semibold text-slate-900">
              Research, simplified.
            </p>

            <p className="mt-1 text-xs leading-5 text-slate-500">
              Explore scientific evidence and your own research documents.
            </p>
          </div>

          <p className="mt-4 text-center text-[10px] text-slate-400">
            SciFact Research Assistant
          </p>
        </div>
      </aside>
    </>
  )
}

function Header({ page, setPage, onMenu, apiStatus }) {
  const titles = {
    research: 'Research Assistant',
    documents: 'PDF Library',
    search: 'Document Search',
    'pdf-qa': 'PDF Question Answering',
    status: 'API Status',
    settings: 'Settings',
  }

  return (
    <header className="sticky top-0 z-30 flex h-[76px] items-center justify-between border-b border-slate-200 bg-white/90 px-4 backdrop-blur-xl sm:px-7 lg:px-9">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenu}
          aria-label="Open navigation"
          className="rounded-xl p-2 text-slate-600 hover:bg-slate-100 lg:hidden"
        >
          <Menu size={21} />
        </button>

        <div>
          <p className="text-sm font-semibold text-slate-900">
            {titles[page]}
          </p>

          <p className="mt-0.5 hidden text-xs text-slate-500 sm:block">
            SciFact / {titles[page]}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 sm:flex">
          <span
            className={`h-2 w-2 rounded-full ${
              apiStatus === 'ok'
                ? 'bg-emerald-500'
                : apiStatus === 'error'
                  ? 'bg-red-500'
                  : 'bg-amber-400'
            }`}
          />

          <span className="text-xs font-medium text-slate-600">
            {apiStatus === 'ok'
              ? 'API connected'
              : apiStatus === 'error'
                ? 'API unavailable'
                : 'Checking API'}
          </span>
        </div>

        <AppButton onClick={() => setPage('research')}>
          <Plus size={16} />
          <span className="hidden sm:inline">New question</span>
          <span className="sm:hidden">New</span>
        </AppButton>
      </div>
    </header>
  )
}

function ResearchHome({ onAsk }) {
  const [question, setQuestion] = useState('')

  function submitQuestion(event) {
    event.preventDefault()

    if (question.trim().length < 3) return

    onAsk(question.trim())
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-9 pt-5 text-center sm:pt-12">
        <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-indigo-100 bg-white text-indigo-600 shadow-sm">
          <Sparkles size={26} />
        </div>

        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50/70 px-3 py-1.5 text-[11px] font-semibold text-indigo-700">
          SCIENTIFIC RESEARCH WORKSPACE
        </div>

        <h1 className="text-3xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
          What does the research
          <span className="mt-1 block text-indigo-600">
            actually say?
          </span>
        </h1>

        <p className="mx-auto mt-5 max-w-xl text-sm leading-7 text-slate-500 sm:text-base">
          Ask scientific questions, explore research evidence, and get answers
          grounded in retrieved documents.
        </p>
      </div>

      <form
        onSubmit={submitQuestion}
        className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm transition focus-within:border-indigo-300 sm:p-5"
      >
        <label htmlFor="research-question" className="sr-only">
          Enter your scientific research question
        </label>

        <textarea
          id="research-question"
          value={question}
          onChange={(event) =>
            setQuestion(event.target.value.slice(0, 1000))
          }
          minLength={3}
          maxLength={1000}
          rows={3}
          placeholder="Ask a scientific question or explore a research topic..."
          className="w-full resize-y border-0 bg-transparent px-2 py-2 text-sm leading-7 text-slate-800 outline-none placeholder:text-slate-400 sm:text-base"
        />

        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-3">
          <span className="px-2 text-[11px] text-slate-400">
            {question.length}/1000 characters
          </span>

          <AppButton type="submit" disabled={question.trim().length < 3}>
            Ask Research Assistant
            <ArrowUpRight size={16} />
          </AppButton>
        </div>
      </form>

      <div className="mt-9">
        <h2 className="text-sm font-semibold text-slate-900">
          Explore a question
        </h2>

        <p className="mt-1 text-xs text-slate-500">
          Start with an example or write your own.
        </p>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {examples.map(({ title, question: example, icon: Icon }) => (
            <button
              key={title}
              onClick={() => setQuestion(example)}
              className="group rounded-2xl border border-slate-200 bg-white p-4 text-left transition hover:-translate-y-0.5 hover:border-indigo-200 hover:shadow-md sm:p-5"
            >
              <div className="mb-5 flex items-center justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-50 text-indigo-600 group-hover:bg-indigo-50">
                  <Icon size={19} />
                </span>

                <ArrowUpRight
                  size={16}
                  className="text-slate-300 group-hover:text-indigo-600"
                />
              </div>

              <p className="text-sm font-semibold text-slate-800">
                {title}
              </p>

              <p className="mt-2 text-xs leading-6 text-slate-500">
                {example}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function ResearchAnswer({ question, onBack }) {
  const [loading, setLoading] = useState(true)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    async function loadAnswer() {
      try {
        const data = await askResearchQuestion(question)

        if (!cancelled) setResult(data)
      } catch (err) {
        if (!cancelled) setError(getErrorMessage(err))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadAnswer()

    return () => {
      cancelled = true
    }
  }, [question])

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      setError('Could not copy to clipboard. Check browser permissions.')
    }
  }

  return (
    <div className="mx-auto max-w-4xl">
      <button
        onClick={onBack}
        className="mb-6 text-sm font-medium text-indigo-600 hover:text-indigo-800"
      >
        ← Back to research
      </button>

      <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-8">
        <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">
          Research question
        </p>

        <h1 className="mt-3 text-xl font-semibold leading-8 text-slate-900">
          {question}
        </h1>

        {loading && (
          <div className="mt-8 flex items-center gap-3 text-sm text-slate-500">
            <RefreshCw className="animate-spin" size={18} />
            Retrieving research evidence and generating your answer...
          </div>
        )}

        {error && (
          <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}

            <div className="mt-3">
              <AppButton
                variant="secondary"
                onClick={() => window.location.reload()}
              >
                Retry
              </AppButton>
            </div>
          </div>
        )}

        {result && (
          <div className="mt-8 space-y-7">
            <section>
              <div className="flex items-center justify-between gap-3">
                <h2 className="font-semibold text-slate-900">Answer</h2>

                <button
                  onClick={() => copyText(result.answer || '')}
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-indigo-600"
                >
                  <Copy size={14} />
                  Copy answer
                </button>
              </div>

              <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">
                {result.answer}
              </p>
            </section>

            <section className="border-t border-slate-100 pt-6">
              <h2 className="font-semibold text-slate-900">
                Research sources ({result.sources?.length ?? 0})
              </h2>

              {(result.sources ?? []).map((source, index) => (
                <article
                  key={`${source.document_id ?? 'source'}-${index}`}
                  className="mt-3 rounded-xl border border-slate-200 p-4"
                >
                  <p className="text-sm font-semibold text-slate-800">
                    {source.source_label ||
                      source.title ||
                      `Source ${index + 1}`}
                  </p>

                  {source.title && (
                    <p className="mt-1 text-xs text-slate-500">
                      {source.title}
                      {source.retrieval_rank != null
                        ? ` · Rank ${source.retrieval_rank}`
                        : ''}
                    </p>
                  )}

                  <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-600">
                    {source.evidence}
                  </p>

                  <button
                    onClick={() => copyText(source.evidence || '')}
                    className="mt-3 flex items-center gap-1.5 text-xs text-indigo-600 hover:text-indigo-800"
                  >
                    <Copy size={13} />
                    Copy evidence
                  </button>
                </article>
              ))}
            </section>

            <section className="rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-600">
              <p>
                <strong>Model:</strong> {result.model || 'Not provided'}
              </p>
              <p>
                <strong>Retrieval method:</strong>{' '}
                {result.retrieval_method || 'Not provided'}
              </p>
              <p>
                <strong>Citation validation:</strong>{' '}
                {JSON.stringify(
                  result.citation_validation ?? 'Not provided',
                )}
              </p>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}

function DocumentLibrary() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)
  const fileInputRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function loadDocuments() {
      try {
        const data = await getDocuments()

        if (!cancelled) {
          const items = Array.isArray(data)
            ? data
            : data.documents ?? data.items ?? []

          setDocuments(items)
          setError('')
        }
      } catch (err) {
        if (!cancelled) setError(getErrorMessage(err))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadDocuments()

    return () => {
      cancelled = true
    }
  }, [refreshKey])

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    event.target.value = ''

    if (!file) return

    setError('')
    setMessage('')

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file.')
      return
    }

    if (!file.size) {
      setError('The selected file is empty.')
      return
    }

    if (file.size > 15 * 1024 * 1024) {
      setError('The PDF must be no larger than 15 MB.')
      return
    }

    setUploading(true)

    try {
      const result = await uploadDocument(file)

      setMessage(
        result.message || result.status || `Upload processed: ${file.name}`,
      )

      setRefreshKey((key) => key + 1)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="text-3xl font-semibold tracking-tight text-slate-950">
        Your research library
      </h1>

      <p className="mt-3 text-sm leading-6 text-slate-500">
        Upload scientific papers, index their contents, and search their
        evidence.
      </p>

      <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
        <div className="flex flex-col items-center text-center">
          <UploadCloud size={34} className="text-indigo-600" />

          <h2 className="mt-4 text-lg font-semibold text-slate-900">
            Upload a research paper
          </h2>

          <p className="mt-2 text-sm text-slate-500">
            PDF format only · Maximum file size 15 MB
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden"
            onChange={handleUpload}
          />

          <AppButton
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadCloud size={17} />
            {uploading ? 'Uploading and indexing...' : 'Choose PDF'}
          </AppButton>
        </div>
      </div>

      {message && (
        <p className="mt-5 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
          {message}
        </p>
      )}

      {error && (
        <p className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </p>
      )}

      <div className="mt-9">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">
            Indexed documents
          </h2>

          <AppButton
            variant="secondary"
            disabled={loading}
            onClick={() => {
              setLoading(true)
              setRefreshKey((key) => key + 1)
            }}
          >
            <RefreshCw size={15} />
            Refresh
          </AppButton>
        </div>

        {loading ? (
          <p className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
            Loading documents...
          </p>
        ) : documents.length === 0 ? (
          !error && (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center">
              <Files size={28} className="mx-auto text-slate-400" />

              <p className="mt-3 text-sm font-medium text-slate-800">
                No documents to display
              </p>

              <p className="mt-2 text-xs text-slate-500">
                Upload a PDF or check your backend response.
              </p>
            </div>
          )
        ) : (
          <div className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200 bg-white">
            {documents.map((doc, index) => (
              <div
                key={doc.document_id ?? doc.id ?? index}
                className="flex items-start gap-3 p-4"
              >
                <Files size={20} className="text-indigo-600" />

                <div className="min-w-0 flex-1">
                  <p className="break-words text-sm font-medium text-slate-800">
                    {doc.filename ??
                      doc.original_filename ??
                      doc.title ??
                      doc.document_id ??
                      doc.id ??
                      `Document ${index + 1}`}
                  </p>

                  <p className="mt-1 break-all text-xs text-slate-500">
                    {doc.document_id ?? doc.id ?? 'ID not provided'}
                  </p>
                </div>

                <CheckCircle2 size={18} className="text-emerald-600" />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function DocumentSearch() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(3)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function submit(event) {
    event.preventDefault()

    if (query.trim().length < 3) return

    setLoading(true)
    setError('')
    setResults(null)

    try {
      const data = await searchDocuments(query.trim(), topK)
      setResults(data)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const items = Array.isArray(results)
    ? results
    : results?.results ?? results?.documents ?? results?.items ?? []

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl font-semibold text-slate-950">
        Document Search
      </h1>

      <p className="mt-3 text-sm text-slate-500">
        Retrieve evidence from uploaded PDFs without requiring answer
        generation.
      </p>

      <form
        onSubmit={submit}
        className="mt-7 rounded-2xl border border-slate-200 bg-white p-4 sm:p-6"
      >
        <label
          htmlFor="doc-search"
          className="text-sm font-medium text-slate-800"
        >
          Search your documents
        </label>

        <input
          id="doc-search"
          value={query}
          onChange={(event) =>
            setQuery(event.target.value.slice(0, 1000))
          }
          minLength={3}
          maxLength={1000}
          placeholder="e.g. cardiovascular disease"
          className="mt-3 w-full rounded-xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-indigo-400"
        />

        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Results

            <select
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2"
            >
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <AppButton
            type="submit"
            disabled={loading || query.trim().length < 3}
          >
            <Search size={16} />
            {loading ? 'Searching...' : 'Search documents'}
          </AppButton>
        </div>
      </form>

      {error && (
        <p className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-700">
          {error}
        </p>
      )}

      {loading && (
        <p className="mt-6 animate-pulse text-sm text-slate-500">
          Retrieving evidence...
        </p>
      )}

      {results !== null && !loading && (
        <section className="mt-8">
          <h2 className="font-semibold text-slate-900">Search results</h2>

          {items.length === 0 ? (
            <p className="mt-4 rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
              No results found, or the response schema needs to be checked.
            </p>
          ) : (
            items.map((item, index) => (
              <article
                key={item.chunk_id ?? item.document_id ?? item.id ?? index}
                className="mt-3 rounded-xl border border-slate-200 bg-white p-5"
              >
                <p className="text-sm font-semibold text-slate-800">
                  {item.title ??
                    item.filename ??
                    item.source_label ??
                    `Result ${index + 1}`}
                </p>

                <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-600">
                  {item.evidence ??
                    item.text ??
                    item.content ??
                    item.chunk_text ??
                    JSON.stringify(item, null, 2)}
                </p>

                {item.score != null && (
                  <p className="mt-3 text-xs text-slate-400">
                    Score: {item.score}
                  </p>
                )}
              </article>
            ))
          )}
        </section>
      )}
    </div>
  )
}

function PdfQuestionAnswering() {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(3)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function submit(event) {
    event.preventDefault()

    if (question.trim().length < 3) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await askDocuments(question.trim(), topK)
      setResult(data)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl font-semibold text-slate-950">
        Ask your documents
      </h1>

      <p className="mt-3 text-sm leading-6 text-slate-500">
        Ask a question about the scientific PDFs indexed by your backend.
      </p>

      <form
        onSubmit={submit}
        className="mt-7 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7"
      >
        <label
          htmlFor="pdf-question"
          className="text-sm font-medium text-slate-800"
        >
          Your question
        </label>

        <textarea
          id="pdf-question"
          value={question}
          onChange={(event) =>
            setQuestion(event.target.value.slice(0, 1000))
          }
          minLength={3}
          maxLength={1000}
          rows={4}
          placeholder="What are the main findings in these papers?"
          className="mt-3 w-full resize-y rounded-xl border border-slate-200 p-4 text-sm leading-6 outline-none focus:border-indigo-400"
        />

        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Evidence count

            <select
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2"
            >
              {[1, 2, 3].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <AppButton
            type="submit"
            disabled={loading || question.trim().length < 3}
          >
            <MessageSquareText size={16} />
            {loading ? 'Working...' : 'Ask documents'}
          </AppButton>
        </div>
      </form>

      {error && (
        <p className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-700">
          {error}
        </p>
      )}

      {loading && (
        <p className="mt-6 animate-pulse text-sm text-slate-500">
          Searching document evidence...
        </p>
      )}

      {result && (
        <div className="mt-7 space-y-4">
          <section className="rounded-2xl border border-slate-200 bg-white p-6">
            <h2 className="font-semibold text-slate-900">
              Answer / response
            </h2>

            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">
              {typeof result.answer === 'string'
                ? result.answer
                : JSON.stringify(result, null, 2)}
            </p>
          </section>

          {Array.isArray(result.sources) && (
            <section className="rounded-2xl border border-slate-200 bg-white p-6">
              <h2 className="font-semibold text-slate-900">
                Supporting sources
              </h2>

              {result.sources.map((source, index) => (
                <article
                  key={`${source.document_id ?? 'source'}-${index}`}
                  className="mt-4 border-t border-slate-100 pt-4"
                >
                  <p className="text-sm font-semibold text-slate-800">
                    {source.title ??
                      source.source_label ??
                      `Source ${index + 1}`}
                  </p>

                  <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-600">
                    {source.evidence ?? ''}
                  </p>
                </article>
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  )
}

function ApiStatus() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastChecked, setLastChecked] = useState(null)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let cancelled = false

    async function loadHealth() {
      try {
        const response = await apiRequest('/health')

        if (!cancelled) {
          setData(response)
          setLastChecked(new Date())
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setError(getErrorMessage(err))
          setData(null)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadHealth()

    return () => {
      cancelled = true
    }
  }, [refreshKey])

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-3xl font-semibold text-slate-950">
        API Status
      </h1>

      <p className="mt-3 text-sm text-slate-500">
        Live connectivity to your FastAPI backend.
      </p>

      <div className="mt-7 rounded-2xl border border-slate-200 bg-white p-6">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Activity className="text-indigo-600" size={24} />

            <div>
              <p className="font-semibold text-slate-900">
                Backend health
              </p>

              <p className="mt-1 text-xs text-slate-500">
                {loading
                  ? 'Checking...'
                  : data?.status ?? (error ? 'Unavailable' : 'Unknown')}
              </p>
            </div>
          </div>

          <AppButton
            variant="secondary"
            disabled={loading}
            onClick={() => {
              setLoading(true)
              setRefreshKey((key) => key + 1)
            }}
          >
            <RefreshCw size={15} />
            Refresh
          </AppButton>
        </div>

        {error && (
          <p className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-700">
            {error}
          </p>
        )}

        {data && (
          <pre className="mt-5 overflow-auto rounded-xl bg-slate-50 p-4 text-xs leading-6 text-slate-700">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}

        {lastChecked && (
          <p className="mt-4 text-xs text-slate-400">
            Last checked: {lastChecked.toLocaleTimeString()}
          </p>
        )}
      </div>
    </div>
  )
}

function SettingsPage() {
  return (
    <div className="mx-auto flex min-h-[50vh] max-w-2xl flex-col items-center justify-center text-center">
      <Settings2 size={34} className="mb-5 text-indigo-600" />

      <h1 className="text-2xl font-semibold text-slate-950">
        Application settings
      </h1>

      <p className="mt-3 text-sm leading-7 text-slate-500">
        Configuration options will be added as the application develops.
      </p>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('research')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    let cancelled = false

    async function checkApi() {
      try {
        const data = await apiRequest('/health')

        if (!cancelled) {
          setApiStatus(data.status === 'ok' ? 'ok' : 'error')
        }
      } catch {
        if (!cancelled) setApiStatus('error')
      }
    }

    checkApi()

    return () => {
      cancelled = true
    }
  }, [])

  function navigate(nextPage) {
    setPage(nextPage)
    setSubmittedQuestion('')
  }

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar
          page={page}
          setPage={navigate}
          open={sidebarOpen}
          close={() => setSidebarOpen(false)}
        />

        <div className="min-w-0 flex-1">
          <Header
            page={page}
            setPage={navigate}
            onMenu={() => setSidebarOpen(true)}
            apiStatus={apiStatus}
          />

          <main className="min-h-[calc(100vh-145px)] px-4 py-6 sm:px-7 sm:py-9 lg:px-10 lg:py-10">
            {page === 'research' &&
              (submittedQuestion ? (
                <ResearchAnswer
                  question={submittedQuestion}
                  onBack={() => setSubmittedQuestion('')}
                />
              ) : (
                <ResearchHome onAsk={setSubmittedQuestion} />
              ))}

            {page === 'documents' && <DocumentLibrary />}

            {page === 'search' && <DocumentSearch />}

            {page === 'pdf-qa' && <PdfQuestionAnswering />}

            {page === 'status' && <ApiStatus />}

            {page === 'settings' && <SettingsPage />}
          </main>

          <footer className="border-t border-slate-200/80 px-5 py-5 text-center text-[11px] text-slate-400">
            SciFact Research Assistant · Scientific evidence, made accessible.
          </footer>
        </div>
      </div>
    </div>
  )
}