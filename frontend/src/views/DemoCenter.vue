<template>
  <div class="demo-page">
    <div class="page-header">
      <h2 class="page-title">功能演示</h2>
      <p class="page-desc">
        通过<strong>高亮引导（Tour）</strong>浏览各模块界面说明，并可一键载入<strong>示例数据</strong>体验单样本与批量检测（调用真实后端接口）。
      </p>
    </div>

    <el-alert type="info" show-icon :closable="false" class="demo-tip">
      <template #title>使用说明</template>
      全流程演示从首页开始，每页最后一步进入下一模块；分模块「界面引导」仅浏览当前页。单样本「示例+引导」会先自动检测再高亮结果区。
    </el-alert>

    <el-card shadow="never" class="hero-card">
      <div class="hero-inner">
        <div>
          <h3>一键全流程</h3>
          <p class="hero-desc">按侧栏顺序依次展示：首页 → 检测 → 预警 → 模型 → 训练 → 性能 → 技术亮点。</p>
        </div>
        <el-button type="primary" size="large" :icon="VideoPlay" @click="runFullTour">
          开始全流程引导
        </el-button>
      </div>
    </el-card>

    <div class="section-title">分模块演示</div>
    <div class="demo-grid">
      <el-card v-for="card in demoCards" :key="card.path" shadow="hover" class="demo-card">
        <div class="card-title-row">
          <el-icon :size="22" color="#165DFF"><component :is="card.icon" /></el-icon>
          <span>{{ card.title }}</span>
        </div>
        <p class="card-desc">{{ card.desc }}</p>
        <div class="card-actions">
          <el-button type="primary" plain size="small" @click="goPageTour(card.path)">界面引导</el-button>
          <el-button
            v-if="card.extra === 'single-sample'"
            type="success"
            plain
            size="small"
            @click="runSingleWithSample"
          >
            示例 + 引导
          </el-button>
          <el-button
            v-if="card.extra === 'batch-sample'"
            type="success"
            plain
            size="small"
            @click="runBatchSample"
          >
            示例批量检测
          </el-button>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { VideoPlay, HomeFilled, Search, Upload, Bell, Cpu, RefreshRight, DataAnalysis, Star } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { GUIDED_STEPS_BY_PATH } from '@/demo/guidedFlow'
import { guidedDemo, startFullGuidedTour, stopGuidedTour } from '@/demo/guidedDemoState'

const router = useRouter()

const DEMO_SINGLE_KEY = 'phish_demo_single'
const DEMO_BATCH_KEY = 'phish_demo_batch'

const demoCards = [
  { path: '/home', title: '首页', desc: '系统概览、统计卡片与核心功能入口。', icon: HomeFilled },
  {
    path: '/single-detection',
    title: '单样本检测',
    desc: '文本/URL、关联链接与多源分析结果展示。',
    icon: Search,
    extra: 'single-sample',
  },
  {
    path: '/batch-detection',
    title: '批量检测',
    desc: 'CSV 解析、批量请求与结果表。',
    icon: Upload,
    extra: 'batch-sample',
  },
  { path: '/alerts', title: '预警中心', desc: '筛选、列表与预警处理入口。', icon: Bell },
  { path: '/models', title: '模型管理', desc: 'BERT-TextCNN 版本与热重载。', icon: Cpu },
  { path: '/training', title: '增量训练', desc: '训练参数与任务提交界面。', icon: RefreshRight },
  { path: '/performance', title: '性能数据', desc: '趋势与版本指标对比。', icon: DataAnalysis },
  { path: '/tech-highlights', title: '技术亮点', desc: '架构与工程化说明文档化展示。', icon: Star },
]

function runFullTour() {
  stopGuidedTour()
  startFullGuidedTour(router)
  ElMessage.success('已开始全流程演示，请跟随高亮与说明操作')
}

function goPageTour(path) {
  stopGuidedTour()
  router.push(path).then(() => {
    const steps = GUIDED_STEPS_BY_PATH[path]
    if (!steps?.length) {
      ElMessage.warning('该页面暂无引导步骤')
      return
    }
    guidedDemo.mode = 'page'
    guidedDemo.tourCurrent = 0
    setTimeout(() => {
      guidedDemo.tourOpen = true
    }, 400)
  })
}

function runSingleWithSample() {
  stopGuidedTour()
  const payload = {
    inputType: 'text',
    text: '【安全中心】您的账户存在异常登录，请立即点击链接验证身份：http://verify-bank-fake-example.com/login',
    url: 'http://verify-bank-fake-example.com/login',
    scenario: 'sms',
  }
  sessionStorage.setItem(DEMO_SINGLE_KEY, JSON.stringify(payload))
  router.push('/single-detection')
  ElMessage.info('已跳转单样本页，将自动填入示例并检测，随后开启界面引导')
}

function runBatchSample() {
  stopGuidedTour()
  sessionStorage.setItem(DEMO_BATCH_KEY, '1')
  router.push('/batch-detection')
  ElMessage.info('已跳转批量页，将自动载入演示样本并请求检测')
}
</script>

<style scoped>
.demo-page {
  max-width: 1200px;
}
.page-header {
  margin-bottom: 20px;
}
.page-title {
  font-size: 22px;
  font-weight: 600;
  color: #1d2129;
  margin-bottom: 8px;
}
.page-desc {
  font-size: 14px;
  color: #86909c;
  line-height: 1.6;
}
.demo-tip {
  margin-bottom: 20px;
}
.hero-card {
  margin-bottom: 24px;
  border-radius: 8px;
}
.hero-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
}
.hero-inner h3 {
  margin: 0 0 8px;
  font-size: 18px;
  color: #1d2129;
}
.hero-desc {
  margin: 0;
  font-size: 14px;
  color: #86909c;
  max-width: 640px;
}
.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #1d2129;
  margin-bottom: 14px;
}
.demo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}
.demo-card {
  border-radius: 8px;
}
.card-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  margin-bottom: 10px;
  color: #1d2129;
}
.card-desc {
  font-size: 13px;
  color: #86909c;
  line-height: 1.5;
  min-height: 44px;
  margin: 0 0 14px;
}
.card-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
</style>

