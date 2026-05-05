<template>
  <div class="tech-page">
    <div id="demo-tech-header" class="page-header">
      <h2 class="page-title">技术亮点</h2>
      <p class="page-desc">深入解析系统核心技术架构与创新设计</p>
    </div>

    <div class="highlight-section">
      <div class="highlight-main">
        <div class="highlight-left">
          <div class="tag-badge">核心架构</div>
          <h3 class="highlight-title">BERT-TextCNN 混合模型</h3>
          <p class="highlight-desc">
            系统采用BERT与TextCNN的混合架构作为核心识别模型。
            BERT负责全局语义理解，TextCNN专注于局部n-gram特征提取；
            URL与网络行为信息用于输入解析、记录、预警和辅助解释。
          </p>
          <div class="highlight-tags">
            <el-tag type="primary" size="small">深度语义理解</el-tag>
            <el-tag type="success" size="small">局部特征捕获</el-tag>
            <el-tag type="warning" size="small">优势互补</el-tag>
          </div>
        </div>
        <div class="highlight-right">
          <div class="arch-diagram">
            <div class="arch-block input-block">
              <el-icon :size="20" color="#165DFF"><Document /></el-icon>
              <span>文本/URL输入</span>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-row">
              <div class="arch-block bert-block">
                <el-icon :size="20" color="#165DFF"><Cpu /></el-icon>
                <span>BERT</span>
                <small>语义特征</small>
              </div>
              <div class="arch-block textcnn-block">
                <el-icon :size="20" color="#00B42A"><Connection /></el-icon>
                <span>TextCNN</span>
                <small>局部特征</small>
              </div>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-block fusion-block">
              <el-icon :size="20" color="#FF7D00"><Merge /></el-icon>
              <span>特征融合</span>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-block output-block">
              <el-icon :size="20" color="#722ED1"><Finished /></el-icon>
              <span>分类输出</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="tech-grid">
      <div class="tech-card" v-for="tech in techList" :key="tech.title">
        <div class="tech-icon" :style="{ background: tech.bgColor }">
          <el-icon :size="28" :color="tech.color"><component :is="tech.icon" /></el-icon>
        </div>
        <h4 class="tech-title">{{ tech.title }}</h4>
        <p class="tech-desc">{{ tech.desc }}</p>
        <div class="tech-points">
          <div class="point" v-for="point in tech.points" :key="point">
            <el-icon :size="12" :color="tech.color"><Check /></el-icon>
            <span>{{ point }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="section-title">
      <el-icon color="#165DFF"><SetUp /></el-icon>
      <span>工程优化</span>
    </div>
    <div class="optim-row">
      <div class="optim-card" v-for="opt in optimList" :key="opt.title">
        <div class="optim-header">
          <el-icon :size="22" :color="opt.color"><component :is="opt.icon" /></el-icon>
          <h4>{{ opt.title }}</h4>
        </div>
        <p>{{ opt.desc }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const techList = ref([
  {
    title: '多源信息分析',
    desc: '解析文本语义、URL结构和网络行为信息，核心分类由BERT-TextCNN完成',
    icon: 'Merge',
    color: '#165DFF',
    bgColor: '#E8F3FF',
    points: ['BERT-TextCNN核心判别', 'URL结构特征提取', '网络行为辅助分析']
  },
  {
    title: '实时预警机制',
    desc: '置信度高于阈值时自动触发预警，提供攻击类型和特征详情',
    icon: 'Bell',
    color: '#F53F3F',
    bgColor: '#FFECE8',
    points: ['阈值自动触发', '攻击类型识别', '特征详情报告']
  },
  {
    title: '增量训练框架',
    desc: '支持新样本收集、标注、特征提取和模型更新的完整闭环',
    icon: 'RefreshRight',
    color: '#00B42A',
    bgColor: '#E8FFEA',
    points: ['新样本自动标注', '特征自动提取', '模型热更新']
  },
  {
    title: '多场景适配',
    desc: '针对短信、邮件、链接等不同场景优化检测策略',
    icon: 'Switch',
    color: '#FF7D00',
    bgColor: '#FFF7E8',
    points: ['短信场景优化', '邮件场景优化', '链接场景优化']
  }
])

const optimList = ref([
  { title: '模型量化', desc: '支持模型权重压缩与推理加速，为轻量化部署提供基础', icon: 'Lightning', color: '#165DFF' },
  { title: '异步特征收集', desc: '采用异步请求、连接池和缓存机制，提高网络特征收集稳定性', icon: 'Connection', color: '#00B42A' },
  { title: '动态批处理', desc: '根据系统资源自动调整批处理大小，平衡吞吐量与延迟', icon: 'SetUp', color: '#FF7D00' },
  { title: '结构化日志', desc: '多级别日志管理、日志持久化与轮转策略，便于问题定位与系统监控', icon: 'Document', color: '#722ED1' }
])
</script>

<style scoped>
.tech-page {
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

.highlight-section {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 28px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
  margin-bottom: 24px;
}

.highlight-main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  align-items: center;
}

.tag-badge {
  display: inline-block;
  background: #E8F3FF;
  color: #165DFF;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
}

.highlight-title {
  font-size: 22px;
  font-weight: 700;
  color: #1D2129;
  margin-bottom: 12px;
}

.highlight-desc {
  font-size: 14px;
  color: #4E5969;
  line-height: 1.8;
  margin-bottom: 16px;
}

.highlight-tags {
  display: flex;
  gap: 8px;
}

.arch-diagram {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.arch-block {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  min-width: 140px;
}

.arch-block small {
  font-size: 11px;
  font-weight: 400;
  color: #86909C;
  margin-left: 4px;
}

.input-block {
  background: #F2F3F5;
  color: #4E5969;
}

.bert-block {
  background: #E8F3FF;
  color: #165DFF;
}

.textcnn-block {
  background: #E8FFEA;
  color: #00B42A;
}

.fusion-block {
  background: #FFF7E8;
  color: #FF7D00;
}

.output-block {
  background: #F5E8FF;
  color: #722ED1;
}

.arch-row {
  display: flex;
  gap: 20px;
}

.arch-arrow {
  font-size: 18px;
  color: #C9CDD4;
  font-weight: 700;
}

.tech-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}

.tech-card {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.tech-icon {
  width: 48px;
  height: 48px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 14px;
}

.tech-title {
  font-size: 16px;
  font-weight: 600;
  color: #1D2129;
  margin-bottom: 8px;
}

.tech-desc {
  font-size: 13px;
  color: #86909C;
  line-height: 1.6;
  margin-bottom: 14px;
}

.tech-points {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.point {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #4E5969;
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

.optim-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.optim-card {
  background: #FFFFFF;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.optim-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.optim-header h4 {
  font-size: 14px;
  font-weight: 600;
  color: #1D2129;
}

.optim-card p {
  font-size: 13px;
  color: #86909C;
  line-height: 1.6;
}
</style>
