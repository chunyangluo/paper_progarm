<template>
  <div class="home-page">
    <div class="page-header">
      <h2 class="page-title">系统概览</h2>
      <p class="page-desc">基于BERT-TextCNN混合模型的多模态网络钓鱼智能识别系统</p>
    </div>

    <div id="demo-home-stats" class="stats-row">
      <div class="stat-card" v-for="stat in statsData" :key="stat.label">
        <div class="stat-icon" :style="{ background: stat.bgColor }">
          <el-icon :size="24" :color="stat.color"><component :is="stat.icon" /></el-icon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ stat.value }}</div>
          <div class="stat-label">{{ stat.label }}</div>
        </div>
      </div>
    </div>

    <div class="section-title">
      <el-icon color="#165DFF"><Grid /></el-icon>
      <span>核心功能</span>
    </div>
    <div id="demo-home-features" class="feature-grid">
      <div class="feature-card" v-for="feature in features" :key="feature.title" @click="router.push(feature.route)">
        <div class="feature-icon-wrap" :style="{ background: feature.bgColor }">
          <el-icon :size="28" :color="feature.color"><component :is="feature.icon" /></el-icon>
        </div>
        <div class="feature-content">
          <h3 class="feature-title">{{ feature.title }}</h3>
          <p class="feature-desc">{{ feature.desc }}</p>
        </div>
      </div>
    </div>

    <div class="section-title">
      <el-icon color="#165DFF"><TrendCharts /></el-icon>
      <span>性能指标</span>
    </div>
    <div class="performance-row">
      <div class="perf-card" v-for="perf in perfData" :key="perf.label">
        <div class="perf-ring" :style="{ '--progress': perf.value, '--color': perf.color }">
          <span class="perf-value">{{ perf.value }}%</span>
        </div>
        <div class="perf-label">{{ perf.label }}</div>
      </div>
    </div>

    <div class="section-title">
      <el-icon color="#165DFF"><MagicStick /></el-icon>
      <span>技术亮点</span>
    </div>
    <div class="highlight-row">
      <div class="highlight-card" v-for="hl in highlights" :key="hl.title">
        <el-icon :size="20" :color="hl.color"><component :is="hl.icon" /></el-icon>
        <div class="highlight-text">
          <h4>{{ hl.title }}</h4>
          <p>{{ hl.desc }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { modelApi, systemApi } from '@/api'

const router = useRouter()

const statsData = ref([
  { label: '总检测量', value: '-', icon: 'TrendCharts', color: '#165DFF', bgColor: '#E8F3FF' },
  { label: '钓鱼占比', value: '-', icon: 'Grid', color: '#00B42A', bgColor: '#E8FFEA' },
  { label: '活跃模型', value: '-', icon: 'Timer', color: '#FF7D00', bgColor: '#FFF7E8' },
  { label: '未处理预警', value: '-', icon: 'Connection', color: '#722ED1', bgColor: '#F5E8FF' }
])

const features = ref([
  { title: '多模态钓鱼检测', desc: '支持文本、URL和网络行为信息输入解析，核心判别采用BERT-TextCNN模型', icon: 'View', color: '#165DFF', bgColor: '#E8F3FF', route: '/single-detection' },
  { title: '实时预警系统', desc: '置信度高于0.7时自动生成预警，包含攻击类型、置信度和特征详情', icon: 'Bell', color: '#F53F3F', bgColor: '#FFECE8', route: '/alerts' },
  { title: '可视化分析', desc: '展示检测统计、趋势数据和BERT-TextCNN模型版本指标', icon: 'PieChart', color: '#00B42A', bgColor: '#E8FFEA', route: '/performance' },
  { title: '增量训练', desc: '支持收集新样本标注、自动特征提取和模型更新部署', icon: 'RefreshRight', color: '#FF7D00', bgColor: '#FFF7E8', route: '/training' },
  { title: '模型管理', desc: 'BERT-TextCNN模型版本、性能监控和热更新管理', icon: 'Cpu', color: '#722ED1', bgColor: '#F5E8FF', route: '/models' },
  { title: '批量检测', desc: '支持text/url列CSV批量导入，高效处理文本或链接样本', icon: 'Files', color: '#165DFF', bgColor: '#E8F3FF', route: '/batch-detection' },
  { title: '功能演示', desc: '一键全流程引导、分模块界面说明与示例数据体验', icon: 'VideoPlay', color: '#722ED1', bgColor: '#F5E8FF', route: '/demo' }
])

const perfData = ref([
  { label: '准确率', value: 0, color: '#165DFF' },
  { label: '精确率', value: 0, color: '#00B42A' },
  { label: '召回率', value: 0, color: '#FF7D00' },
  { label: 'F1值', value: 0, color: '#722ED1' }
])

const highlights = ref([
  { title: 'BERT语义理解', desc: '利用BERT预训练模型深度理解中文文本语义特征', icon: 'Cpu', color: '#165DFF' },
  { title: 'TextCNN局部捕获', desc: 'TextCNN捕获文本局部n-gram特征，与BERT形成互补', icon: 'Connection', color: '#00B42A' },
  { title: '多源信息分析', desc: '文本、URL、网络行为信息用于辅助解释、记录与预警分析', icon: 'Merge', color: '#FF7D00' },
  { title: '模型量化部署', desc: '支持模型量化压缩，降低内存占用，加速推理部署', icon: 'Lightning', color: '#722ED1' }
])

onMounted(async () => {
  try {
    const [statsRes, versionsRes] = await Promise.all([
      systemApi.stats(),
      modelApi.list(),
    ])
    if (statsRes) {
      statsData.value[0].value = `${statsRes.total_detections || 0}`
      const total = statsRes.total_detections || 0
      const phishing = statsRes.phishing_count || 0
      statsData.value[1].value = total > 0 ? `${((phishing / total) * 100).toFixed(1)}%` : '0.0%'
      statsData.value[2].value = statsRes.active_model || '-'
      statsData.value[3].value = `${statsRes.unhandled_alerts || 0}`
    }

    const activeVersion = (Array.isArray(versionsRes) ? versionsRes : []).find(v => v.is_active)
    if (activeVersion) {
      perfData.value[0].value = Number(((activeVersion.accuracy || 0) * 100).toFixed(1))
      perfData.value[1].value = Number(((activeVersion.precision || 0) * 100).toFixed(1))
      perfData.value[2].value = Number(((activeVersion.recall || 0) * 100).toFixed(1))
      perfData.value[3].value = Number(((activeVersion.f1_score || 0) * 100).toFixed(1))
    }
  } catch {}
})
</script>

<style scoped>
.home-page {
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

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}

.stat-card {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.stat-value {
  font-size: 22px;
  font-weight: 700;
  color: #1D2129;
}

.stat-label {
  font-size: 13px;
  color: #86909C;
  margin-top: 2px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 16px;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}

.feature-card {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  gap: 16px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
  transition: box-shadow 0.2s;
  cursor: pointer;
}

.feature-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.feature-icon-wrap {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.feature-title {
  font-size: 15px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 6px;
}

.feature-desc {
  font-size: 13px;
  color: #86909C;
  line-height: 1.5;
}

.performance-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}

.perf-card {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 24px 16px;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.perf-ring {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: conic-gradient(var(--color) calc(var(--progress) * 1%), #F2F3F5 0);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 12px;
  position: relative;
}

.perf-ring::before {
  content: '';
  width: 76px;
  height: 76px;
  background: #FFFFFF;
  border-radius: 50%;
  position: absolute;
}

.perf-value {
  position: relative;
  z-index: 1;
  font-size: 16px;
  font-weight: 700;
  color: #1D2129;
}

.perf-label {
  font-size: 14px;
  color: #4E5969;
  font-weight: 500;
}

.highlight-row {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.highlight-card {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: flex-start;
  gap: 14px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.highlight-card h4 {
  font-size: 14px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 4px;
}

.highlight-card p {
  font-size: 13px;
  color: #86909C;
  line-height: 1.5;
}
</style>
