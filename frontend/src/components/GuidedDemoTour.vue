<template>
  <el-tour
    v-model="guidedDemo.tourOpen"
    v-model:current="guidedDemo.tourCurrent"
    type="primary"
    :show-close="true"
    :mask="{ color: 'rgba(0,0,0,0.42)' }"
    :gap="{ offset: 8, radius: 4 }"
    :z-index="4000"
    :target-area-clickable="true"
    @close="onTourClose"
    @finish="onTourFinish"
  >
    <el-tour-step
      v-for="(step, i) in activeSteps"
      :key="`${route.path}-${i}-${guidedDemo.mode}`"
      :target="step.target"
      :title="step.title"
      :description="step.description"
      :placement="step.placement || 'bottom'"
      :next-button-props="nextButtonProps(i)"
    />
  </el-tour>
</template>

<script setup>
import { computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { GUIDED_STEPS_BY_PATH, FULL_FLOW_ORDER, nextModuleButtonLabel } from '@/demo/guidedFlow'
import { guidedDemo, stopGuidedTour } from '@/demo/guidedDemoState'

const route = useRoute()
const router = useRouter()

const activeSteps = computed(() => {
  if (!guidedDemo.tourOpen) return []
  if (guidedDemo.mode === 'single' && route.path === '/single-detection') {
    return GUIDED_STEPS_BY_PATH['/single-detection'] || []
  }
  if (guidedDemo.mode === 'full' || guidedDemo.mode === 'page') {
    return GUIDED_STEPS_BY_PATH[route.path] || []
  }
  return []
})

watch(
  () => route.path,
  (newPath, oldPath) => {
    if (!guidedDemo.tourOpen) return
    if (guidedDemo.mode === 'page' && oldPath !== undefined && newPath !== oldPath) {
      stopGuidedTour()
      return
    }
    if (guidedDemo.mode === 'full') {
      guidedDemo.tourCurrent = 0
    }
  },
)

watch(
  () => guidedDemo.tourOpen,
  (open) => {
    if (open) guidedDemo.tourCurrent = 0
  },
)

function nextButtonProps(index) {
  const steps = activeSteps.value
  const last = index === steps.length - 1
  if (!last) return undefined
  if (guidedDemo.mode === 'full') {
    return { children: nextModuleButtonLabel(route.path) }
  }
  return { children: '完成' }
}

function onTourClose() {
  stopGuidedTour()
}

function onTourFinish() {
  if (guidedDemo.mode === 'full') {
    const order = FULL_FLOW_ORDER
    const idx = order.indexOf(route.path)
    if (idx < 0 || idx >= order.length - 1) {
      stopGuidedTour()
      ElMessage.success('全流程演示已结束')
      return
    }
    guidedDemo.tourOpen = false
    const nextPath = order[idx + 1]
    router.push(nextPath).then(() => {
      guidedDemo.tourCurrent = 0
      setTimeout(() => {
        guidedDemo.tourOpen = true
      }, 400)
    })
    return
  }
  if (guidedDemo.mode === 'page') {
    stopGuidedTour()
    ElMessage.success('本模块引导已结束')
    return
  }
  if (guidedDemo.mode === 'single') {
    stopGuidedTour()
    ElMessage.success('单样本页引导已结束')
    return
  }
  stopGuidedTour()
}
</script>
