import { reactive } from 'vue'
import { FULL_FLOW_ORDER } from './guidedFlow'

export const guidedDemo = reactive({
  /** idle | full | single */
  mode: 'idle',
  tourOpen: false,
  tourCurrent: 0,
})

export function stopGuidedTour() {
  guidedDemo.mode = 'idle'
  guidedDemo.tourOpen = false
  guidedDemo.tourCurrent = 0
}

/** 启动跨页全流程（从首页开始） */
export function startFullGuidedTour(router) {
  guidedDemo.mode = 'full'
  guidedDemo.tourCurrent = 0
  guidedDemo.tourOpen = false
  router.push('/home').then(() => {
    setTimeout(() => {
      guidedDemo.tourOpen = true
    }, 350)
  })
}

/** 仅当前页：单样本检测引导（需已在 /single-detection） */
export function startSinglePageTour(router) {
  guidedDemo.mode = 'single'
  guidedDemo.tourCurrent = 0
  guidedDemo.tourOpen = false
  const go = () => {
    setTimeout(() => {
      guidedDemo.tourOpen = true
    }, 350)
  }
  if (router.currentRoute.value.path === '/single-detection') {
    go()
  } else {
    router.push('/single-detection').then(go)
  }
}

/** 已在单样本页且完成填样时，仅打开本页引导 */
export function openSingleGuidedTour() {
  guidedDemo.mode = 'single'
  guidedDemo.tourCurrent = 0
  setTimeout(() => {
    guidedDemo.tourOpen = true
  }, 400)
}

export function isFullFlowPath(path) {
  return FULL_FLOW_ORDER.includes(path)
}
