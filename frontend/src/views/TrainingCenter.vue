<template>
  <div class="training-page">
    <div id="demo-training-header" class="page-header">
      <h2 class="page-title">增量训练</h2>
      <p class="page-desc">训练任务配置与执行控制，当前系统以 BERT-TextCNN 作为核心识别模型</p>
    </div>

    <div class="training-container">
      <el-card shadow="never" class="config-card">
        <template #header>
          <div class="card-header">
            <el-icon color="#165DFF"><Setting /></el-icon>
            <span>训练配置</span>
          </div>
        </template>

        <el-form :model="trainForm" label-width="100px">
          <el-form-item label="模型类型">
            <el-select v-model="trainForm.model_type" style="width: 100%">
              <el-option label="BERT-TextCNN（核心识别模型）" value="bert_textcnn" />
            </el-select>
          </el-form-item>

          <el-form-item label="训练轮数">
            <el-input-number v-model="trainForm.epochs" :min="1" :max="100" style="width: 100%" />
          </el-form-item>

          <el-form-item label="学习率">
            <el-input-number v-model="trainForm.learning_rate" :min="0.000001" :max="0.01" :step="0.000001" :precision="6" style="width: 100%" />
          </el-form-item>

          <el-form-item label="批大小">
            <el-input-number v-model="trainForm.batch_size" :min="1" :max="256" style="width: 100%" />
          </el-form-item>

          <el-form-item label="数据集路径">
            <el-input v-model="trainForm.dataset_path" placeholder="留空使用默认数据集" />
          </el-form-item>

          <el-form-item>
            <el-button type="primary" @click="startTraining" :loading="starting" style="width: 100%">
              <el-icon v-if="!starting"><VideoPlay /></el-icon>
              {{ starting ? '提交中...' : '开始训练' }}
            </el-button>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card shadow="never" class="tasks-card">
        <template #header>
          <div class="card-header-row">
            <div class="card-header">
              <el-icon color="#165DFF"><List /></el-icon>
              <span>训练任务</span>
            </div>
            <el-button size="small" @click="loadTasks" :icon="Refresh">刷新</el-button>
          </div>
        </template>

        <el-table :data="tasks" stripe style="width: 100%" v-loading="loading">
          <el-table-column prop="task_id" label="任务ID" width="180" show-overflow-tooltip />
          <el-table-column prop="model_type" label="模型类型" width="130" />
          <el-table-column prop="status" label="状态" width="100" align="center">
            <template #default="{ row }">
              <el-tag :type="statusType(row.status)" size="small">{{ statusLabel(row.status) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="current_epoch" label="进度" width="120" align="center">
            <template #default="{ row }">
              {{ row.current_epoch || 0 }} / {{ row.epochs || '-' }}
            </template>
          </el-table-column>
          <el-table-column prop="best_accuracy" label="最佳准确率" width="120" align="center">
            <template #default="{ row }">
              {{ row.best_accuracy ? (row.best_accuracy * 100).toFixed(2) + '%' : '-' }}
            </template>
          </el-table-column>
          <el-table-column prop="best_f1_score" label="最佳F1" width="100" align="center">
            <template #default="{ row }">
              {{ row.best_f1_score ? (row.best_f1_score * 100).toFixed(2) + '%' : '-' }}
            </template>
          </el-table-column>
          <el-table-column prop="error_message" label="错误信息" min-width="200" show-overflow-tooltip />
          <el-table-column prop="created_at" label="创建时间" width="170" show-overflow-tooltip />
          <el-table-column label="操作" width="100" align="center" fixed="right">
            <template #default="{ row }">
              <el-button
                v-if="row.status === 'pending' || row.status === 'running'"
                type="danger"
                size="small"
                @click="cancelTask(row)"
              >
                取消
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { trainingApi } from '@/api'

const trainForm = reactive({
  model_type: 'bert_textcnn',
  epochs: 10,
  learning_rate: 0.00001,
  batch_size: 16,
  dataset_path: ''
})

const tasks = ref([])
const loading = ref(false)
const starting = ref(false)

const statusType = (s) => ({ pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }[s] || '')
const statusLabel = (s) => ({ pending: '等待中', running: '训练中', completed: '已完成', failed: '失败' }[s] || s)

const loadTasks = async () => {
  loading.value = true
  try {
    const res = await trainingApi.listTasks()
    tasks.value = Array.isArray(res) ? res : []
  } catch {
    tasks.value = []
  } finally {
    loading.value = false
  }
}

const startTraining = async () => {
  starting.value = true
  try {
    const data = { ...trainForm }
    if (!data.dataset_path) delete data.dataset_path
    const res = await trainingApi.start(data)
    ElMessage.success(`训练任务已创建: ${res.task_id}`)
    loadTasks()
  } catch {
    // Global axios interceptor handles request errors.
  } finally {
    starting.value = false
  }
}

const cancelTask = async (row) => {
  try {
    await ElMessageBox.confirm('确定要取消此训练任务吗？', '确认', { type: 'warning' })
    await trainingApi.cancelTask(row.task_id)
    ElMessage.success('任务已取消')
    loadTasks()
  } catch {}
}

onMounted(loadTasks)
</script>

<style scoped>
.training-page {
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

.training-container {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 20px;
}

.config-card, .tasks-card {
  border-radius: 8px;
}

.config-card :deep(.el-card__header),
.tasks-card :deep(.el-card__header) {
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

.card-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
