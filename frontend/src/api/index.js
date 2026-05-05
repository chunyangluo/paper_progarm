import axios from 'axios'
import { ElMessage } from 'element-plus'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

apiClient.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

apiClient.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    const status = error.response?.status
    const shouldToast = !error.config?.skipGlobalErrorHandler && status !== 404
    if (shouldToast) {
      const message = error.response?.data?.detail || error.response?.data?.error || error.message || '请求失败'
      ElMessage.error(message)
      error.__handledByGlobal = true
    }
    return Promise.reject(error)
  }
)

export const detectionApi = {
  detectSingle(data) {
    return apiClient.post('/api/v1/detection/single', data)
  },
  detectBatch(data) {
    return apiClient.post('/api/v1/detection/batch', data)
  },
  detectSms(data) {
    return apiClient.post('/api/v1/detection/sms', data)
  },
  detectEmail(data) {
    return apiClient.post('/api/v1/detection/email', data)
  },
  detectLink(data) {
    return apiClient.post('/api/v1/detection/link', data)
  },
}

export const alertApi = {
  list(params = {}) {
    return apiClient.get('/api/v1/alerts', { params })
  },
  getUnhandledCount() {
    return apiClient.get('/api/v1/alerts/unhandled-count')
  },
  get(alertId) {
    return apiClient.get(`/api/v1/alerts/${alertId}`)
  },
  handle(alertId, data) {
    return apiClient.post(`/api/v1/alerts/${alertId}/handle`, data)
  },
}

export const sampleApi = {
  list(params = {}) {
    return apiClient.get('/api/v1/samples', { params })
  },
  create(data) {
    return apiClient.post('/api/v1/samples', data)
  },
  createBatch(data) {
    return apiClient.post('/api/v1/samples/batch', data)
  },
  get(sampleId) {
    return apiClient.get(`/api/v1/samples/${sampleId}`)
  },
  delete(sampleId) {
    return apiClient.delete(`/api/v1/samples/${sampleId}`)
  },
}

export const modelApi = {
  list(params = {}) {
    return apiClient.get('/api/v1/models', { params })
  },
  getActive(params = {}) {
    return apiClient.get('/api/v1/models/active', { params })
  },
  getInfo() {
    return apiClient.get('/api/v1/models/info')
  },
  activate(data) {
    return apiClient.post('/api/v1/models/activate', data)
  },
  reload(modelType) {
    return apiClient.post(`/api/v1/models/reload/${modelType}`)
  },
  setActive(modelType) {
    return apiClient.post(`/api/v1/models/set-active/${modelType}`)
  },
  getPerformance(params = {}) {
    return apiClient.get('/api/v1/models/performance', { params })
  },
}

export const trainingApi = {
  start(data) {
    return apiClient.post('/api/v1/training/start', data)
  },
  listTasks(params = {}) {
    return apiClient.get('/api/v1/training/tasks', { params })
  },
  getTask(taskId) {
    return apiClient.get(`/api/v1/training/tasks/${taskId}`)
  },
  cancelTask(taskId) {
    return apiClient.post(`/api/v1/training/tasks/${taskId}/cancel`)
  },
}

export const systemApi = {
  health() {
    return apiClient.get('/api/v1/system/health')
  },
  stats() {
    return apiClient.get('/api/v1/system/stats')
  },
  trends(params = {}) {
    return apiClient.get('/api/v1/system/trends', { params })
  },
  resources() {
    return apiClient.get('/api/v1/system/resources')
  },
  createBackup() {
    return apiClient.post('/api/v1/system/backup')
  },
  listBackups() {
    return apiClient.get('/api/v1/system/backups')
  },
}

export default apiClient
