
import { apiRequest } from './client'

export function askResearchQuestion(question) {
  return apiRequest('/api/v1/ask', {
    method: 'POST',
    body: JSON.stringify({ question }),
  })
}
