import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('@/layout/MainLayout.vue'),
    redirect: '/home',
    children: [
      {
        path: 'home',
        name: 'Home',
        component: () => import('@/views/Home.vue'),
        meta: { title: '首页' }
      },
      {
        path: 'single-detection',
        name: 'SingleDetection',
        component: () => import('@/views/SingleDetection.vue'),
        meta: { title: '单样本检测' }
      },
      {
        path: 'batch-detection',
        name: 'BatchDetection',
        component: () => import('@/views/BatchDetection.vue'),
        meta: { title: '批量检测' }
      },
      {
        path: 'alerts',
        name: 'AlertCenter',
        component: () => import('@/views/AlertCenter.vue'),
        meta: { title: '预警中心' }
      },
      {
        path: 'performance',
        name: 'Performance',
        component: () => import('@/views/Performance.vue'),
        meta: { title: '性能数据' }
      },
      {
        path: 'models',
        name: 'ModelManagement',
        component: () => import('@/views/ModelManagement.vue'),
        meta: { title: '模型管理' }
      },
      {
        path: 'training',
        name: 'TrainingCenter',
        component: () => import('@/views/TrainingCenter.vue'),
        meta: { title: '增量训练' }
      },
      {
        path: 'tech-highlights',
        name: 'TechHighlights',
        component: () => import('@/views/TechHighlights.vue'),
        meta: { title: '技术亮点' }
      },
      {
        path: 'demo',
        name: 'DemoCenter',
        component: () => import('@/views/DemoCenter.vue'),
        meta: { title: '功能演示' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
