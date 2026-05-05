<template>
  <div class="batch-page">
    <div class="page-header">
      <h2 class="page-title">批量检测</h2>
      <p class="page-desc">上传CSV文件批量检测，支持 text、url、scenario 列；URL-only样本也可检测</p>
    </div>

    <div class="batch-container">
      <el-card id="demo-batch-upload" shadow="never" class="upload-card">
        <template #header>
          <div class="card-header">
            <el-icon color="#165DFF"><Upload /></el-icon>
            <span>文件上传</span>
          </div>
        </template>

        <el-upload
          ref="uploadRef"
          class="upload-area"
          drag
          action="#"
          :auto-upload="false"
          accept=".csv"
          :on-change="handleFileChange"
          :limit="1"
          :on-exceed="handleExceed"
        >
          <el-icon :size="48" color="#C9CDD4"><UploadFilled /></el-icon>
          <div class="upload-text">将CSV文件拖拽到此处，或<em>点击上传</em></div>
          <div class="upload-tip">仅支持.csv格式文件；至少包含 text 或 url 列，可选 scenario 列</div>
        </el-upload>

        <div id="demo-batch-actions" class="upload-actions">
          <el-button type="primary" @click="handleBatchDetect" :loading="processing" :disabled="!fileSelected">
            <el-icon v-if="!processing"><Search /></el-icon>
            {{ processing ? '处理中...' : '开始批量检测' }}
          </el-button>
          <el-button @click="handleReset" :disabled="processing">重置</el-button>
        </div>

        <div v-if="processing" class="processing-status">
          <el-icon class="is-loading" :size="18" color="#165DFF"><Loading /></el-icon>
          <span>处理中，请稍候...</span>
          <el-progress :percentage="processProgress" :stroke-width="8" style="margin-top: 8px" />
        </div>

        <div v-if="processComplete" class="complete-status">
          <el-icon :size="18" color="#00B42A"><CircleCheckFilled /></el-icon>
          <span>处理完成！共检测 <strong>{{ resultData.length }}</strong> 条数据</span>
        </div>
      </el-card>

      <el-card v-if="resultData.length > 0" shadow="never" class="result-card">
        <template #header>
          <div class="card-header">
            <el-icon color="#165DFF"><Document /></el-icon>
            <span>检测结果</span>
            <el-tag type="info" size="small" style="margin-left: 8px">共 {{ resultData.length }} 条</el-tag>
          </div>
        </template>

        <el-table :data="resultData" stripe style="width: 100%" max-height="420" size="default">
          <el-table-column type="index" label="序号" width="60" align="center" />
          <el-table-column prop="text" label="文本内容" min-width="280" show-overflow-tooltip />
          <el-table-column prop="url" label="URL" min-width="220" show-overflow-tooltip />
          <el-table-column prop="scenario" label="场景类型" width="100" align="center">
            <template #default="{ row }">
              <el-tag size="small" :type="scenarioTagType(row.scenario)">{{ scenarioLabel(row.scenario) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="result" label="检测结果" width="120" align="center">
            <template #default="{ row }">
              <el-tag :type="row.result === '钓鱼' ? 'danger' : 'success'" size="small">
                {{ row.result }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="confidence" label="置信度" width="180" align="center">
            <template #default="{ row }">
              <el-progress
                :percentage="row.confidence"
                :color="row.result === '钓鱼' ? '#F53F3F' : '#00B42A'"
                :stroke-width="8"
                :text-inside="true"
              />
            </template>
          </el-table-column>
          <el-table-column prop="attackType" label="攻击类型" width="120" align="center" />
        </el-table>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { detectionApi } from '@/api'

const uploadRef = ref(null)
const fileSelected = ref(false)
const processing = ref(false)
const processComplete = ref(false)
const processProgress = ref(0)
const resultData = ref([])
const parsedItems = ref([])

const DEMO_BATCH_KEY = 'phish_demo_batch'

async function runDemoBatchFromCenter() {
  parsedItems.value = [
    {
      text: '您的积分即将过期请尽快兑换奖品',
      url: 'http://reward-phish-demo-example.net/claim',
      scenario: 'sms',
      use_cache: false,
    },
    {
      text: '今日快递已放到小区门卫，请凭取件码领取',
      url: '',
      scenario: 'general',
      use_cache: false,
    },
  ]
  fileSelected.value = true
  processComplete.value = false
  resultData.value = []
  ElMessage.info('已载入演示样本，正在请求批量检测…')
  await handleBatchDetect()
}

onMounted(async () => {
  if (!sessionStorage.getItem(DEMO_BATCH_KEY)) return
  sessionStorage.removeItem(DEMO_BATCH_KEY)
  await runDemoBatchFromCenter()
})

const scenarioTagType = (s) => {
  const map = { sms: 'warning', email: '', link: 'info', general: 'success' }
  return map[s] || ''
}

const scenarioLabel = (s) => {
  const map = { sms: '短信', email: '邮件', link: '链接', general: '通用' }
  return map[s] || s || '通用'
}

const parseCsvLine = (line) => {
  const values = []
  let current = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"' && line[i + 1] === '"') {
      current += '"'
      i++
    } else if (ch === '"') {
      inQuotes = !inQuotes
    } else if (ch === ',' && !inQuotes) {
      values.push(current.trim())
      current = ''
    } else {
      current += ch
    }
  }
  values.push(current.trim())
  return values
}

const normalizeScenario = (raw) => {
  const val = (raw || '').trim().toLowerCase()
  if (['sms', '短信'].includes(val)) return 'sms'
  if (['email', '邮件', 'mail'].includes(val)) return 'email'
  if (['link', '链接', 'url'].includes(val)) return 'link'
  return 'general'
}

const handleFileChange = async (file) => {
  if (!file.name.endsWith('.csv')) {
    ElMessage.error('仅支持CSV格式文件')
    uploadRef.value?.clearFiles()
    fileSelected.value = false
    return
  }
  try {
    const text = await file.raw.text()
    const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
    if (lines.length <= 1) {
      ElMessage.warning('CSV文件内容为空或仅包含表头')
      fileSelected.value = false
      return
    }
    const headers = parseCsvLine(lines[0]).map(h => h.toLowerCase())
    const textIdx = headers.findIndex(h => h === 'text')
    const urlIdx = headers.findIndex(h => h === 'url')
    const scenarioIdx = headers.findIndex(h => h === 'scenario')
    if (textIdx === -1 && urlIdx === -1) {
      ElMessage.error('CSV必须至少包含 text 或 url 列')
      fileSelected.value = false
      return
    }

    parsedItems.value = lines.slice(1).map(line => parseCsvLine(line)).map(cols => ({
      text: textIdx >= 0 ? (cols[textIdx] || '') : (cols[urlIdx] || ''),
      url: urlIdx >= 0 ? (cols[urlIdx] || '') : '',
      scenario: scenarioIdx >= 0 ? normalizeScenario(cols[scenarioIdx]) : 'general',
      use_cache: true,
    })).filter(item => (item.text && item.text.trim().length > 0) || (item.url && item.url.trim().length > 0))

    if (parsedItems.value.length === 0) {
      ElMessage.warning('没有可用于检测的有效 text/url 数据')
      fileSelected.value = false
      return
    }
    fileSelected.value = true
    processComplete.value = false
    resultData.value = []
    ElMessage.success(`文件解析成功，共 ${parsedItems.value.length} 条`)
  } catch {
    ElMessage.error('CSV解析失败，请检查文件编码与格式')
    fileSelected.value = false
  }
}

const handleExceed = () => {
  ElMessage.warning('仅支持上传一个文件，请先移除已选文件')
}

const handleBatchDetect = async () => {
  if (!fileSelected.value || parsedItems.value.length === 0) {
    ElMessage.warning('请先上传并解析CSV文件')
    return
  }

  processing.value = true
  processComplete.value = false
  processProgress.value = 10
  resultData.value = []

  try {
    const res = await detectionApi.detectBatch({
      items: parsedItems.value,
    })
    processProgress.value = 90
    resultData.value = (res.results || []).map((item, idx) => ({
      text: parsedItems.value[idx]?.text || '',
      url: parsedItems.value[idx]?.url || '',
      scenario: item.scenario,
      result: item.prediction,
      confidence: Math.round((item.confidence || 0) * 100),
      attackType: item.is_phishing ? '疑似钓鱼' : '无',
    }))
    processProgress.value = 100
    processComplete.value = true
    ElMessage.success(`批量检测完成：钓鱼 ${res.phishing_count || 0} 条，正常 ${res.normal_count || 0} 条`)
  } catch {
    processProgress.value = 0
  } finally {
    processing.value = false
  }
}

const handleReset = () => {
  uploadRef.value?.clearFiles()
  fileSelected.value = false
  processing.value = false
  processComplete.value = false
  processProgress.value = 0
  resultData.value = []
  parsedItems.value = []
}
</script>

<style scoped>
.batch-page {
  max-width: 1640px;
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  font-size: 22px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 6px;
}

.page-desc {
  font-size: 14px;
  color: #86909C;
}

.upload-card, .result-card {
  border-radius: 8px;
  margin-bottom: 20px;
}

.upload-card :deep(.el-card__header),
.result-card :deep(.el-card__header) {
  padding: 16px 20px;
  border-bottom: 1px solid #F2F3F5;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
  color: #1D2129;
}

.upload-area :deep(.el-upload-dragger) {
  padding: 40px 0;
  border-radius: 8px;
}

.upload-text {
  font-size: 14px;
  color: #4E5969;
  margin-top: 12px;
}

.upload-text em {
  color: #165DFF;
  font-style: normal;
}

.upload-tip {
  font-size: 12px;
  color: #C9CDD4;
  margin-top: 8px;
}

.upload-actions {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

.processing-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 16px;
  padding: 12px 16px;
  background: #E8F3FF;
  border-radius: 6px;
  font-size: 14px;
  color: #165DFF;
  flex-wrap: wrap;
}

.complete-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 16px;
  padding: 12px 16px;
  background: #E8FFEA;
  border-radius: 6px;
  font-size: 14px;
  color: #00B42A;
}

.complete-status strong {
  color: #00B42A;
}
</style>
