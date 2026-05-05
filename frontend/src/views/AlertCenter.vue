<template>
  <div class="alert-page">
    <div id="demo-alerts-header" class="page-header">
      <h2 class="page-title">预警中心</h2>
      <p class="page-desc">实时预警信息推送与可视化展示，支持预警处理与追踪</p>
    </div>

    <div class="filter-bar">
      <el-radio-group v-model="filter" @change="loadAlerts">
        <el-radio-button value="all">全部</el-radio-button>
        <el-radio-button value="unhandled">未处理</el-radio-button>
        <el-radio-button value="critical">严重</el-radio-button>
        <el-radio-button value="high">高危</el-radio-button>
        <el-radio-button value="medium">中危</el-radio-button>
      </el-radio-group>
      <el-button @click="loadAlerts" :icon="Refresh">刷新</el-button>
    </div>

    <el-card id="demo-alerts-table" shadow="never" class="alert-card">
      <el-table :data="alerts" stripe style="width: 100%" max-height="600" v-loading="loading">
        <el-table-column prop="alert_id" label="预警ID" width="180" show-overflow-tooltip />
        <el-table-column prop="severity" label="级别" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="severityType(row.severity)" size="small">{{ severityLabel(row.severity) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100" align="center">
          <template #default="{ row }">
            <span :style="{ color: row.confidence > 0.9 ? '#F53F3F' : row.confidence > 0.7 ? '#FF7D00' : '#4E5969' }">
              {{ (row.confidence * 100).toFixed(1) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="content" label="预警内容" min-width="280" show-overflow-tooltip />
        <el-table-column prop="scenario" label="场景" width="80" align="center">
          <template #default="{ row }">
            {{ scenarioLabel(row.scenario) }}
          </template>
        </el-table-column>
        <el-table-column prop="is_handled" label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.is_handled ? 'success' : 'danger'" size="small">
              {{ row.is_handled ? '已处理' : '未处理' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="时间" width="170" show-overflow-tooltip />
        <el-table-column label="操作" width="120" align="center" fixed="right">
          <template #default="{ row }">
            <el-button v-if="!row.is_handled" type="primary" size="small" @click="handleAlert(row)">
              处理
            </el-button>
            <el-tag v-else type="info" size="small">已处理</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="handleDialogVisible" title="处理预警" width="500px">
      <el-form :model="handleForm" label-width="80px">
        <el-form-item label="处理人">
          <el-input v-model="handleForm.handler" placeholder="请输入处理人姓名" />
        </el-form-item>
        <el-form-item label="处理备注">
          <el-input v-model="handleForm.note" type="textarea" :rows="4" placeholder="请输入处理备注" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="handleDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitHandle" :loading="submitting">确认处理</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { alertApi } from '@/api'

const filter = ref('all')
const alerts = ref([])
const loading = ref(false)
const handleDialogVisible = ref(false)
const submitting = ref(false)
const currentAlert = ref(null)

const handleForm = reactive({
  handler: '',
  note: ''
})

const severityType = (s) => ({ critical: 'danger', high: 'warning', medium: '', low: 'info' }[s] || '')
const severityLabel = (s) => ({ critical: '严重', high: '高危', medium: '中危', low: '低危' }[s] || s)
const scenarioLabel = (s) => ({ sms: '短信', email: '邮件', link: '链接', general: '通用' }[s] || s)

const loadAlerts = async () => {
  loading.value = true
  try {
    const params = {}
    if (filter.value === 'unhandled') params.unhandled_only = true
    else if (['critical', 'high', 'medium'].includes(filter.value)) params.severity = filter.value

    const res = await alertApi.list(params)
    alerts.value = Array.isArray(res) ? res : []
  } catch {
    alerts.value = []
  } finally {
    loading.value = false
  }
}

const handleAlert = (row) => {
  currentAlert.value = row
  handleForm.handler = ''
  handleForm.note = ''
  handleDialogVisible.value = true
}

const submitHandle = async () => {
  if (!handleForm.handler) {
    ElMessage.warning('请输入处理人姓名')
    return
  }
  submitting.value = true
  try {
    await alertApi.handle(currentAlert.value.alert_id, handleForm)
    ElMessage.success('预警已处理')
    handleDialogVisible.value = false
    loadAlerts()
  } catch {
    // Global axios interceptor handles request errors.
  } finally {
    submitting.value = false
  }
}

onMounted(loadAlerts)
</script>

<style scoped>
.alert-page {
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

.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.alert-card {
  border-radius: 8px;
}
</style>
