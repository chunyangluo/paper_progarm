<template>
  <div class="performance-page">
    <div id="demo-performance-header" class="page-header">
      <h2 class="page-title">性能数据</h2>
      <p class="page-desc">实时展示检测统计、趋势数据与BERT-TextCNN模型版本指标</p>
    </div>

    <el-row :gutter="16" class="summary-row">
      <el-col :span="6"><el-card shadow="never" class="summary-card"><div class="k">总检测量</div><div class="v">{{ stats.total_detections }}</div></el-card></el-col>
      <el-col :span="6"><el-card shadow="never" class="summary-card"><div class="k">钓鱼占比</div><div class="v">{{ phishingRate }}%</div></el-card></el-col>
      <el-col :span="6"><el-card shadow="never" class="summary-card"><div class="k">平均置信度</div><div class="v">{{ (stats.avg_confidence * 100).toFixed(1) }}%</div></el-card></el-col>
      <el-col :span="6"><el-card shadow="never" class="summary-card"><div class="k">未处理预警</div><div class="v">{{ stats.unhandled_alerts }}</div></el-card></el-col>
    </el-row>

    <div class="section-title">
      <el-icon color="#165DFF"><Histogram /></el-icon>
      <span>BERT-TextCNN版本指标对比</span>
    </div>
    <el-card shadow="never" class="chart-card">
      <div ref="metricChartRef" class="chart-container"></div>
    </el-card>

    <div class="section-title">
      <el-icon color="#165DFF"><TrendCharts /></el-icon>
      <span>近30天检测趋势</span>
    </div>
    <el-card shadow="never" class="chart-card">
      <div ref="trendChartRef" class="chart-container"></div>
    </el-card>

    <div class="section-title">
      <el-icon color="#165DFF"><Grid /></el-icon>
      <span>模型版本明细</span>
    </div>
    <el-card shadow="never" class="chart-card">
      <el-table :data="versionRows" stripe v-loading="loading" max-height="420">
        <el-table-column prop="version" label="版本" width="120" />
        <el-table-column prop="model_type" label="模型类型" width="130" />
        <el-table-column prop="accuracy" label="Accuracy" width="120" />
        <el-table-column prop="precision" label="Precision" width="120" />
        <el-table-column prop="recall" label="Recall" width="120" />
        <el-table-column prop="f1_score" label="F1" width="120" />
        <el-table-column prop="auc_score" label="AUC" width="120" />
        <el-table-column prop="is_active" label="活跃" width="90" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import { modelApi, systemApi } from '@/api'

const loading = ref(false)
const stats = reactive({
  total_detections: 0,
  phishing_count: 0,
  normal_count: 0,
  avg_confidence: 0,
  unhandled_alerts: 0,
})
const trends = ref([])
const versions = ref([])
const versionRows = computed(() => (versions.value || []).map(v => ({
  ...v,
  accuracy: v.accuracy != null ? `${(v.accuracy * 100).toFixed(2)}%` : '-',
  precision: v.precision != null ? `${(v.precision * 100).toFixed(2)}%` : '-',
  recall: v.recall != null ? `${(v.recall * 100).toFixed(2)}%` : '-',
  f1_score: v.f1_score != null ? `${(v.f1_score * 100).toFixed(2)}%` : '-',
  auc_score: v.auc_score != null ? `${(v.auc_score * 100).toFixed(2)}%` : '-',
  is_active: v.is_active ? '是' : '否',
})))
const phishingRate = computed(() => {
  if (!stats.total_detections) return '0.0'
  return ((stats.phishing_count / stats.total_detections) * 100).toFixed(1)
})

const metricChartRef = ref(null)
const trendChartRef = ref(null)
let metricChart = null
let trendChart = null

const renderMetricChart = () => {
  if (!metricChartRef.value) return
  if (!metricChart) metricChart = echarts.init(metricChartRef.value)
  const selected = (versions.value || []).filter(v => v.model_type === 'bert_textcnn')
  const labels = selected.map(v => `BERT-TextCNN ${v.version}`)
  metricChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['Accuracy', 'F1', 'AUC'] },
    xAxis: { type: 'category', data: labels },
    yAxis: { type: 'value', min: 0, max: 1 },
    series: [
      { name: 'Accuracy', type: 'bar', data: selected.map(v => v.accuracy ?? 0) },
      { name: 'F1', type: 'bar', data: selected.map(v => v.f1_score ?? 0) },
      { name: 'AUC', type: 'bar', data: selected.map(v => v.auc_score ?? 0) },
    ]
  })
}

const renderTrendChart = () => {
  if (!trendChartRef.value) return
  if (!trendChart) trendChart = echarts.init(trendChartRef.value)
  trendChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['总检测', '钓鱼'] },
    xAxis: { type: 'category', data: trends.value.map(t => t.date) },
    yAxis: { type: 'value' },
    series: [
      { name: '总检测', type: 'line', smooth: true, data: trends.value.map(t => t.total || 0) },
      { name: '钓鱼', type: 'line', smooth: true, data: trends.value.map(t => t.phishing || 0) },
    ]
  })
}

const loadData = async () => {
  loading.value = true
  try {
    const [statsRes, trendRes, versionsRes] = await Promise.all([
      systemApi.stats(),
      systemApi.trends({ days: 30 }),
      modelApi.list(),
    ])
    Object.assign(stats, statsRes || {})
    trends.value = Array.isArray(trendRes) ? trendRes : []
    versions.value = Array.isArray(versionsRes) ? versionsRes : []
    await nextTick()
    renderMetricChart()
    renderTrendChart()
  } finally {
    loading.value = false
  }
}

onMounted(loadData)
onBeforeUnmount(() => {
  metricChart?.dispose()
  trendChart?.dispose()
})
</script>

<style scoped>
.performance-page {
  max-width: 1640px;
}

.summary-row {
  margin-bottom: 16px;
}

.summary-card .k {
  color: #86909C;
  font-size: 13px;
}

.summary-card .v {
  color: #1D2129;
  font-size: 22px;
  font-weight: 700;
  margin-top: 6px;
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

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 16px;
  margin-top: 24px;
}

.section-title:first-of-type {
  margin-top: 0;
}

.chart-card {
  border-radius: 8px;
}

.chart-container {
  width: 100%;
  height: 360px;
}

</style>
