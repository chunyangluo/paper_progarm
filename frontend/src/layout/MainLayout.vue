<template>
  <el-container class="main-layout">
    <el-header class="header">
      <div class="header-left">
        <el-icon :size="24" color="#165DFF"><Monitor /></el-icon>
        <span class="header-title">网络钓鱼智能识别系统</span>
      </div>
      <div class="header-right">
        <span class="header-subtitle">BERT-TextCNN 核心识别 · 多源信息分析</span>
        <el-badge v-if="alertCount > 0" :value="alertCount" :max="99" class="alert-badge">
          <el-icon :size="20" color="#F53F3F" style="cursor:pointer" @click="router.push('/alerts')"><Bell /></el-icon>
        </el-badge>
      </div>
    </el-header>
    <el-container class="body-container">
      <el-aside width="220px" class="sidebar">
        <el-menu
          id="demo-sidebar"
          :default-active="activeMenu"
          class="sidebar-menu"
          @select="handleMenuSelect"
        >
          <el-menu-item-group title="检测中心">
            <el-menu-item index="/home">
              <el-icon><HomeFilled /></el-icon>
              <span>首页</span>
            </el-menu-item>
            <el-menu-item index="/single-detection">
              <el-icon><Search /></el-icon>
              <span>单样本检测</span>
            </el-menu-item>
            <el-menu-item index="/batch-detection">
              <el-icon><Upload /></el-icon>
              <span>批量检测</span>
            </el-menu-item>
          </el-menu-item-group>
          <el-menu-item-group title="预警与管理">
            <el-menu-item index="/alerts">
              <el-icon><Bell /></el-icon>
              <span>预警中心</span>
            </el-menu-item>
            <el-menu-item index="/models">
              <el-icon><Cpu /></el-icon>
              <span>模型管理</span>
            </el-menu-item>
            <el-menu-item index="/training">
              <el-icon><RefreshRight /></el-icon>
              <span>增量训练</span>
            </el-menu-item>
          </el-menu-item-group>
          <el-menu-item-group title="分析">
            <el-menu-item index="/performance">
              <el-icon><DataAnalysis /></el-icon>
              <span>性能数据</span>
            </el-menu-item>
            <el-menu-item index="/tech-highlights">
              <el-icon><Star /></el-icon>
              <span>技术亮点</span>
            </el-menu-item>
          </el-menu-item-group>
          <el-menu-item-group title="演示">
            <el-menu-item index="/demo">
              <el-icon><VideoPlay /></el-icon>
              <span>功能演示</span>
            </el-menu-item>
          </el-menu-item-group>
        </el-menu>
      </el-aside>
      <el-main class="main-content">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
    <GuidedDemoTour />
  </el-container>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { alertApi } from '@/api'
import GuidedDemoTour from '@/components/GuidedDemoTour.vue'

const router = useRouter()
const route = useRoute()

const activeMenu = computed(() => route.path)
const alertCount = ref(0)

let pollTimer = null

const fetchAlertCount = async () => {
  try {
    const res = await alertApi.getUnhandledCount()
    alertCount.value = res.unhandled_count || 0
  } catch {
    alertCount.value = 0
  }
}

const handleMenuSelect = (index) => {
  router.push(index)
}

onMounted(() => {
  fetchAlertCount()
  pollTimer = setInterval(fetchAlertCount, 30000)
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.main-layout {
  width: 100%;
  min-height: 100vh;
  overflow: hidden;
}

.header {
  height: 60px;
  background: #FFFFFF;
  border-bottom: 1px solid #E5E6EB;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
  z-index: 10;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-title {
  font-size: 18px;
  font-weight: 600;
  color: #1D2129;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-subtitle {
  font-size: 13px;
  color: #86909C;
}

.alert-badge {
  margin-left: 8px;
}

.body-container {
  height: calc(100vh - 60px);
}

.sidebar {
  background: #FFFFFF;
  border-right: 1px solid #E5E6EB;
  overflow-y: auto;
}

.sidebar-menu {
  border-right: none;
  padding-top: 8px;
}

.sidebar-menu :deep(.el-menu-item-group__title) {
  font-size: 12px;
  color: #86909C;
  padding: 12px 20px 4px;
}

.sidebar-menu .el-menu-item {
  height: 44px;
  line-height: 44px;
  font-size: 14px;
  color: #4E5969;
  margin: 2px 8px;
  border-radius: 6px;
}

.sidebar-menu .el-menu-item:hover {
  background: #F2F3F5;
  color: #165DFF;
}

.sidebar-menu .el-menu-item.is-active {
  background: #E8F3FF;
  color: #165DFF;
  font-weight: 500;
}

.main-content {
  background: #F5F7FA;
  padding: 20px 24px;
  overflow-y: auto;
  height: calc(100vh - 60px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
