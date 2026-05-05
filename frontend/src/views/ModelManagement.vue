<template>
  <div class="model-page">
    <div id="demo-models-header" class="page-header">
      <h2 class="page-title">模型管理</h2>
      <p class="page-desc">BERT-TextCNN核心识别模型管理；多源信息用于输入解析、记录与辅助分析</p>
    </div>

    <div class="model-info-row">
      <el-card shadow="never" class="info-card">
        <template #header><span class="card-title">当前模型状态</span></template>
        <div class="info-grid">
          <div class="info-item">
            <span class="info-label">活跃模型</span>
            <span class="info-value">{{ modelLabel(modelInfo.active_model) }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">已加载模型</span>
            <span class="info-value">{{ loadedModelLabels }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">运行设备</span>
            <span class="info-value">{{ modelInfo.device || '-' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">缓存大小</span>
            <span class="info-value">{{ modelInfo.cache_size || 0 }}</span>
          </div>
        </div>
      </el-card>

      <el-card shadow="never" class="info-card">
        <template #header><span class="card-title">快捷操作</span></template>
        <div class="action-grid">
          <el-button type="primary" @click="reloadModel('bert_textcnn')" :loading="reloading">
            重载 BERT-TextCNN
          </el-button>
          <el-button @click="setActiveModel('bert_textcnn')">切换 BERT-TextCNN</el-button>
        </div>
        <el-alert
          title="多源信息用于输入解析、记录与辅助分析；系统不再加载单独的多模态神经融合模型。"
          type="info"
          show-icon
          :closable="false"
          class="model-note"
        />
      </el-card>
    </div>

    <el-card shadow="never" class="version-card">
      <template #header>
        <div class="card-header-row">
          <span class="card-title">模型版本列表</span>
          <el-button size="small" @click="loadVersions" :icon="Refresh">刷新</el-button>
        </div>
      </template>

      <el-table :data="versions" stripe style="width: 100%" v-loading="loading">
        <el-table-column prop="version" label="版本号" width="120" />
        <el-table-column prop="model_type" label="模型类型" width="170">
          <template #default="{ row }">{{ modelLabel(row.model_type) }}</template>
        </el-table-column>
        <el-table-column prop="accuracy" label="准确率" width="100" align="center">
          <template #default="{ row }">
            {{ row.accuracy ? (row.accuracy * 100).toFixed(2) + '%' : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="f1_score" label="F1分数" width="100" align="center">
          <template #default="{ row }">
            {{ row.f1_score ? (row.f1_score * 100).toFixed(2) + '%' : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="is_active" label="活跃" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.is_active ? 'success' : 'info'" size="small">
              {{ row.is_active ? '是' : '否' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="is_deployed" label="已部署" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.is_deployed ? 'success' : 'info'" size="small">
              {{ row.is_deployed ? '是' : '否' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
        <el-table-column prop="created_at" label="创建时间" width="170" show-overflow-tooltip />
        <el-table-column label="操作" width="120" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="!row.is_active"
              type="primary"
              size="small"
              @click="activateVersion(row)"
            >
              激活
            </el-button>
            <el-tag v-else type="success" size="small">当前</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { modelApi } from '@/api'

const modelInfo = ref({})
const versions = ref([])
const loading = ref(false)
const reloading = ref(false)

const modelLabel = (modelType) => {
  const map = { bert_textcnn: 'BERT-TextCNN' }
  return map[modelType] || modelType || '-'
}

const loadedModelLabels = computed(() => {
  const loaded = modelInfo.value.loaded_models || []
  return loaded.length ? loaded.map(modelLabel).join(', ') : '-'
})

const loadModelInfo = async () => {
  try {
    const res = await modelApi.getInfo()
    modelInfo.value = res.inference_service || {}
  } catch {}
}

const loadVersions = async () => {
  loading.value = true
  try {
    const res = await modelApi.list()
    versions.value = Array.isArray(res) ? res : []
  } catch {
    versions.value = []
  } finally {
    loading.value = false
  }
}

const reloadModel = async (modelType) => {
  reloading.value = true
  try {
    await modelApi.reload(modelType)
    ElMessage.success(`${modelLabel(modelType)} 模型重载成功`)
    loadModelInfo()
  } catch {
    // Global axios interceptor handles request errors.
  } finally {
    reloading.value = false
  }
}

const setActiveModel = async (modelType) => {
  try {
    await modelApi.setActive(modelType)
    ElMessage.success(`已切换到 ${modelLabel(modelType)}`)
    loadModelInfo()
  } catch {
    // Global axios interceptor handles request errors.
  }
}

const activateVersion = async (row) => {
  try {
    await modelApi.activate({ version: row.version, model_type: row.model_type })
    ElMessage.success('版本已激活')
    loadVersions()
    loadModelInfo()
  } catch {
    // Global axios interceptor handles request errors.
  }
}

onMounted(() => {
  loadModelInfo()
  loadVersions()
})
</script>

<style scoped>
.model-page {
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

.model-info-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}

.info-card, .version-card {
  border-radius: 8px;
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: #1D2129;
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.info-label {
  font-size: 13px;
  color: #86909C;
}

.info-value {
  font-size: 15px;
  font-weight: 600;
  color: #1D2129;
}

.action-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.model-note {
  margin-top: 12px;
}

.card-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
