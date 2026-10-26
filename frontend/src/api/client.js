
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : options.body
          ? { 'Content-Type': 'application/json' }
          : {}),
      ...options.headers,
    },
  })

  const contentType = response.headers.get('content-type') || ''
  const data = contentType.includes('application/json')
    ? await response.json()
    : await response.text()

  if (!response.ok) {
    const detail =
      typeof data === 'object' && data !== null
        ? data.detail || data.message
        : data

    throw new Error(
      typeof detail === 'string'
        ? detail
        : `Request failed with HTTP ${response.status}`,
    )
  }

  return data
}
