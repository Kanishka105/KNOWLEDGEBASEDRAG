
import { apiRequest } from './client'

export function getDocuments() {
  return apiRequest('/api/v1/documents')
}

export function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  return apiRequest('/api/v1/documents/upload', {
    method: 'POST',
    body: formData,
  })
}

export function searchDocuments(query, topK = 3) {
  return apiRequest('/api/v1/documents/search', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  })
}

export function askDocuments(question, topK = 3) {
  return apiRequest('/api/v1/documents/ask', {
    method: 'POST',
    body: JSON.stringify({ question, top_k: topK }),
  })
}
