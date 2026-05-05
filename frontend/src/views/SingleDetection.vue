<template>
  <div class="single-page">
    <div class="page-header">
      <h2 class="page-title">单样本检测</h2>
      <p class="page-desc">输入文本或URL，系统解析多源信息并使用 BERT-TextCNN 核心模型完成判别</p>
    </div>

    <div class="detect-container">
      <div class="input-section">
        <el-card id="demo-single-input" shadow="never" class="input-card">
          <template #header>
            <div class="card-header">
              <el-icon color="#165DFF"><EditPen /></el-icon>
              <span>输入检测内容</span>
            </div>
          </template>

          <el-form label-position="top" :model="form">
            <el-form-item label="输入类型">
              <el-radio-group v-model="form.inputType" @change="handleInputTypeChange">
                <el-radio value="text">文本输入</el-radio>
                <el-radio value="url">URL输入</el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item v-if="form.inputType === 'text'" label="待检测文本">
              <el-input
                v-model="form.text"
                type="textarea"
                :rows="6"
                placeholder="请输入待检测的文本内容，如短信、邮件正文等..."
                maxlength="500"
                show-word-limit
              />
            </el-form-item>

            <el-form-item v-if="form.inputType === 'text'" label="关联URL（可选）">
              <el-input
                v-model="form.url"
                placeholder="可填写短信/邮件中的链接，用于URL与网络行为特征分析"
                clearable
              >
                <template #prefix>
                  <el-icon><Link /></el-icon>
                </template>
              </el-input>
            </el-form-item>

            <el-form-item v-if="form.inputType === 'url'" label="待检测URL">
              <el-input
                v-model="form.url"
                placeholder="请输入待检测的URL地址，如 http://example.com"
                clearable
              >
                <template #prefix>
                  <el-icon><Link /></el-icon>
                </template>
              </el-input>
            </el-form-item>

            <el-form-item label="检测场景">
              <el-select v-model="form.scenario" placeholder="请选择检测场景" style="width: 100%">
                <el-option label="短信场景" value="sms" />
                <el-option label="邮件场景" value="email" />
                <el-option label="链接场景" value="link" />
                <el-option label="通用场景" value="general" />
              </el-select>
            </el-form-item>

            <el-alert
              title="系统支持文本、URL与网络行为信息解析；核心判别模型固定为 BERT-TextCNN。"
              type="info"
              show-icon
              :closable="false"
              class="model-note"
            />

            <el-form-item>
              <el-button type="primary" @click="handleDetect" :loading="detecting" style="width: 100%">
                <el-icon v-if="!detecting"><Search /></el-icon>
                {{ detecting ? '检测中...' : '开始检测' }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </div>

      <div class="result-section">
        <el-card id="demo-single-result" shadow="never" class="result-card">
          <template #header>
            <div class="card-header">
              <el-icon color="#165DFF"><DataLine /></el-icon>
              <span>检测结果</span>
            </div>
          </template>

          <div v-if="!hasResult" class="no-result">
            <el-icon :size="48" color="#C9CDD4"><Warning /></el-icon>
            <p>请输入内容并点击检测按钮</p>
          </div>

          <div v-else class="result-content">
            <div class="result-verdict" :class="result.isPhishing ? 'phishing' : 'safe'">
              <el-icon :size="32">
                <component :is="result.isPhishing ? 'WarningFilled' : 'CircleCheckFilled'" />
              </el-icon>
              <span class="verdict-text">{{ result.isPhishing ? '检测到钓鱼攻击' : '内容安全' }}</span>
            </div>

            <div class="result-details">
              <div class="detail-row">
                <span class="detail-label">置信度</span>
                <div class="detail-bar-wrap">
                  <el-progress
                    :percentage="Math.round(result.confidence * 100)"
                    :color="result.isPhishing ? '#F53F3F' : '#00B42A'"
                    :stroke-width="12"
                    :text-inside="true"
                  />
                </div>
              </div>

              <div class="detail-row">
                <span class="detail-label">检测场景</span>
                <span class="detail-value">{{ scenarioLabel(result.scenario) }}</span>
              </div>

              <div class="detail-row">
                <span class="detail-label">使用模型</span>
                <span class="detail-value">{{ modelLabel(result.model) }}</span>
              </div>

              <div class="detail-row">
                <span class="detail-label">处理时间</span>
                <span class="detail-value">{{ result.processingTime?.toFixed(3) }}s</span>
              </div>

              <div class="detail-row">
                <span class="detail-label">缓存命中</span>
                <span class="detail-value">{{ result.fromCache ? '是' : '否' }}</span>
              </div>
            </div>

            <div v-if="result.details" class="feature-breakdown">
              <h4>多模态信息分析</h4>
              <p class="feature-note">{{ result.details.multimodal_role }}</p>
              <div v-if="result.details.feature_summary" class="summary-grid">
                <div class="summary-item">
                  <span>URL输入</span>
                  <strong>{{ result.details.feature_summary.has_url ? '已提供' : '未提供' }}</strong>
                </div>
                <div class="summary-item">
                  <span>HTTPS</span>
                  <strong>{{ result.details.feature_summary.uses_https ? '是' : '否' }}</strong>
                </div>
                <div class="summary-item">
                  <span>URL长度</span>
                  <strong>{{ result.details.feature_summary.url_length || 0 }}</strong>
                </div>
                <div class="summary-item">
                  <span>域名点数</span>
                  <strong>{{ result.details.feature_summary.domain_dot_count || 0 }}</strong>
                </div>
              </div>
              <div class="feature-items">
                <div class="feature-item">
                  <div class="feat-header">
                    <span class="feat-name">URL特征 ({{ result.details.url_features?.length || 0 }}维)</span>
                  </div>
                  <el-progress
                    :percentage="getUrlFeatureScore()"
                    :color="getUrlFeatureScore() > 70 ? '#F53F3F' : getUrlFeatureScore() > 40 ? '#FF7D00' : '#00B42A'"
                    :show-text="true"
                    :stroke-width="6"
                  />
                </div>
                <div class="feature-item">
                  <div class="feat-header">
                    <span class="feat-name">网络行为特征 ({{ result.details.network_features?.length || 0 }}维)</span>
                  </div>
                  <el-progress
                    :percentage="getNetworkFeatureScore()"
                    :color="getNetworkFeatureScore() > 70 ? '#F53F3F' : getNetworkFeatureScore() > 40 ? '#FF7D00' : '#00B42A'"
                    :show-text="true"
                    :stroke-width="6"
                  />
                </div>
              </div>
            </div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { detectionApi } from '@/api'
import { openSingleGuidedTour } from '@/demo/guidedDemoState'

const DEMO_SINGLE_KEY = 'phish_demo_single'

const form = reactive({
  inputType: 'text',
  text: '',
  url: '',
  scenario: 'general'
})

const detecting = ref(false)
const hasResult = ref(false)
const pendingTourAfterDemo = ref(false)

const result = reactive({
  isPhishing: false,
  confidence: 0,
  scenario: '',
  model: '',
  processingTime: 0,
  fromCache: false,
  details: null
})

const scenarioLabel = (s) => {
  const map = { sms: '短信', email: '邮件', link: '链接', general: '通用' }
  return map[s] || s
}

const modelLabel = (m) => {
  const map = { bert_textcnn: 'BERT-TextCNN 核心识别模型', rule_based: '规则兜底检测' }
  return map[m] || m || '-'
}

const handleInputTypeChange = () => {
  hasResult.value = false
  if (form.inputType === 'url') {
    form.scenario = 'link'
  } else if (form.scenario === 'link') {
    form.scenario = 'general'
  }
}

const getUrlFeatureScore = () => {
  if (!result.details?.url_features) return 0
  const f = result.details.url_features
  const suspicious = f.slice(3).filter(v => v > 0).length
  return Math.min(Math.round(suspicious / 13 * 100), 100)
}

const getNetworkFeatureScore = () => {
  if (!result.details?.network_features) return 0
  const f = result.details.network_features
  const risk = (f[3] || 0) + (f[4] || 0) + (f[7] || 0)
  return Math.min(Math.round(risk / 3 * 100), 100)
}

const handleDetect = async () => {
  const input = form.inputType === 'text' ? form.text : form.url
  if (!input || !input.trim()) {
    ElMessage.warning('请输入待检测内容')
    pendingTourAfterDemo.value = false
    return
  }

  detecting.value = true
  hasResult.value = false

  try {
    const requestData = {
      text: form.inputType === 'text' ? form.text : form.url,
      url: form.inputType === 'url' ? form.url : (form.url || ''),
      scenario: form.scenario,
    }
    const res = await detectionApi.detectSingle(requestData)

    result.isPhishing = res.is_phishing
    result.confidence = res.confidence
    result.scenario = res.scenario
    result.model = res.model
    result.processingTime = res.processing_time
    result.fromCache = res.from_cache
    result.details = res.details
    hasResult.value = true

    if (result.isPhishing) {
      ElMessage.error('检测到钓鱼攻击！请谨慎处理')
    } else {
      ElMessage.success('内容安全，未检测到钓鱼攻击')
    }
    if (pendingTourAfterDemo.value) {
      pendingTourAfterDemo.value = false
      await nextTick()
      openSingleGuidedTour()
    }
  } catch (e) {
    pendingTourAfterDemo.value = false
    // Global axios interceptor handles request errors.
  } finally {
    detecting.value = false
  }
}

onMounted(async () => {
  const raw = sessionStorage.getItem(DEMO_SINGLE_KEY)
  if (!raw) return
  try {
    const p = JSON.parse(raw)
    sessionStorage.removeItem(DEMO_SINGLE_KEY)
    if (p.inputType) form.inputType = p.inputType
    if (p.text != null) form.text = p.text
    if (p.url != null) form.url = p.url
    if (p.scenario) form.scenario = p.scenario
    pendingTourAfterDemo.value = true
    await nextTick()
    await handleDetect()
  } catch {
    sessionStorage.removeItem(DEMO_SINGLE_KEY)
  }
})
</script>

<style scoped>
.single-page {
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

.detect-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.input-card, .result-card {
  border-radius: 8px;
}

.input-card :deep(.el-card__header),
.result-card :deep(.el-card__header) {
  padding: 16px 20px;
  border-bottom: 1px solid #F2F3F5;
}

.input-card :deep(.el-card__body),
.result-card :deep(.el-card__body) {
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
  color: #1D2129;
}

.no-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 400px;
  color: #C9CDD4;
}

.no-result p {
  margin-top: 16px;
  font-size: 14px;
}

.result-content {
  padding: 8px 0;
}

.result-verdict {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 24px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.result-verdict.phishing {
  background: #FFECE8;
  color: #F53F3F;
}

.result-verdict.safe {
  background: #E8FFEA;
  color: #00B42A;
}

.verdict-text {
  font-size: 20px;
  font-weight: 700;
}

.result-details {
  margin-bottom: 24px;
}

.detail-row {
  display: flex;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #F7F8FA;
}

.detail-label {
  width: 80px;
  font-size: 13px;
  color: #86909C;
  flex-shrink: 0;
}

.detail-value {
  font-size: 14px;
  color: #1D2129;
  font-weight: 500;
}

.detail-bar-wrap {
  flex: 1;
}

.model-note {
  margin-bottom: 18px;
}

.feature-breakdown h4 {
  font-size: 14px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 12px;
}

.feature-note {
  margin: -4px 0 12px;
  color: #86909C;
  font-size: 12px;
  line-height: 1.6;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 14px;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 10px;
  background: #F7F8FA;
  border-radius: 6px;
  color: #4E5969;
  font-size: 12px;
}

.summary-item strong {
  color: #1D2129;
}

.feature-items {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.feature-item {
  padding: 8px 0;
}

.feat-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
}

.feat-name {
  font-size: 13px;
  color: #4E5969;
}
</style>
