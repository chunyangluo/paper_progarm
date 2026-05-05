/** 全流程演示的页面顺序（与侧栏功能对应） */
export const FULL_FLOW_ORDER = [
  '/home',
  '/single-detection',
  '/batch-detection',
  '/alerts',
  '/models',
  '/training',
  '/performance',
  '/tech-highlights',
]

/** 各页引导步骤：target 为页面内 DOM id（#xxx） */
export const GUIDED_STEPS_BY_PATH = {
  '/home': [
    {
      target: '#demo-home-stats',
      title: '运行概览',
      description: '展示总检测量、钓鱼占比、当前活跃模型与未处理预警等系统运行指标。',
      placement: 'bottom',
    },
    {
      target: '#demo-home-features',
      title: '核心功能入口',
      description: '由此可进入单样本检测、批量检测、预警、性能与训练等模块；演示模式下可配合高亮引导逐步了解。',
      placement: 'top',
    },
  ],
  '/single-detection': [
    {
      target: '#demo-single-input',
      title: '输入与场景',
      description: '支持文本或 URL、可选关联链接与场景；核心判别固定为 BERT-TextCNN，多源信息用于分析与展示。',
      placement: 'right',
    },
    {
      target: '#demo-single-result',
      title: '检测结果与多源分析',
      description: '查看判别结论、置信度，以及 URL/网络行为等辅助特征与说明文本。',
      placement: 'left',
    },
  ],
  '/batch-detection': [
    {
      target: '#demo-batch-upload',
      title: 'CSV 批量导入',
      description: '支持含 text 或 url 列的 CSV；可混合场景列。演示中心可一键载入示例数据并触发检测。',
      placement: 'bottom',
    },
    {
      target: '#demo-batch-actions',
      title: '批量执行与结果表',
      description: '解析完成后开始批量检测，结果表展示每条样本的判别与置信度。',
      placement: 'top',
    },
  ],
  '/alerts': [
    {
      target: '#demo-alerts-header',
      title: '预警中心',
      description: '按级别与处理状态筛选预警，支持刷新与后续处理流程。',
      placement: 'bottom',
    },
    {
      target: '#demo-alerts-table',
      title: '预警列表',
      description: '展示预警级别、置信度、场景与处理状态；高置信度检测可自动产生预警记录。',
      placement: 'top',
    },
  ],
  '/models': [
    {
      target: '#demo-models-header',
      title: '模型与版本',
      description: '查看 BERT-TextCNN 版本、指标与热重载；生产环境仅挂载该核心识别模型。',
      placement: 'bottom',
    },
  ],
  '/training': [
    {
      target: '#demo-training-header',
      title: '训练与更新',
      description: '配置数据集与超参数，提交训练任务；用于实验或增量更新流程说明。',
      placement: 'bottom',
    },
  ],
  '/performance': [
    {
      target: '#demo-performance-header',
      title: '性能与趋势',
      description: '查看检测统计、趋势曲线及模型版本指标对比（与后端记录一致）。',
      placement: 'bottom',
    },
  ],
  '/tech-highlights': [
    {
      target: '#demo-tech-header',
      title: '技术亮点',
      description: '系统架构、多源信息分析与工程化要点的图文说明，便于答辩与文档对照。',
      placement: 'bottom',
    },
  ],
}

/** 全流程中「当前模块最后一步」按钮文案：进入下一模块 */
export function nextModuleButtonLabel(currentPath) {
  const order = FULL_FLOW_ORDER
  const idx = order.indexOf(currentPath)
  if (idx < 0 || idx >= order.length - 1) return '完成'
  const labels = {
    '/home': '下一步：单样本检测',
    '/single-detection': '下一步：批量检测',
    '/batch-detection': '下一步：预警中心',
    '/alerts': '下一步：模型管理',
    '/models': '下一步：增量训练',
    '/training': '下一步：性能数据',
    '/performance': '下一步：技术亮点',
  }
  return labels[currentPath] || '下一步'
}
