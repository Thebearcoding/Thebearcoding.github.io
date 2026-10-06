// 笔记专题:决定首页专题入口、笔记页的分组顺序和文章页的上一篇/下一篇。
// notes 按推荐阅读顺序排列,第一项是该专题的总览(hub)。
export interface Series {
  key: string;
  name: string;
  desc: string;
  notes: string[];
}

export const SERIES: Series[] = [
  {
    key: 'multimodal',
    name: '多模态算法',
    desc: '从数学与张量预备讲到 Transformer、视觉进入语言模型、视觉定位与视频理解、对比学习与多模态 embedding、SFT/DPO、PPO/GRPO 与多模态 RL、幻觉评测与训练排障,配十四天练习册和复习问答。',
    notes: [
      'multimodal-interview-guide',
      'multimodal-math-prerequisites',
      'transformer-attention-rope-gqa',
      'prefill-decode-video-tokens',
      'vision-video-algorithms',
      'multimodal-grounding',
      'multimodal-video-understanding',
      'contrastive-learning',
      'multimodal-embedding-retrieval',
      'multimodal-sft-lora-dpo',
      'policy-gradient-ppo-grpo',
      'multimodal-rl',
      'multimodal-hallucination-eval',
      'training-memory-debugging',
      'multimodal-models-paper-reading',
      'multimodal-interview-questions',
      'multimodal-fourteen-day-workbook',
      'multimodal-sources-coverage',
    ],
  },
  {
    key: 'agent',
    name: 'Agent 算法',
    desc: '从工具调用协议讲到规划与记忆、RAG、MCP 与多智能体、评测、多轮 Agent RL 与系统设计题,每篇有手算例子与论文出处。',
    notes: [
      'agent-guide',
      'agent-tool-calling',
      'agent-planning-memory',
      'agent-rag',
      'agent-mcp-multiagent',
      'agent-eval',
      'agent-rl-advanced',
      'agent-system-design',
    ],
  },
  {
    key: 'recsys',
    name: '推荐与 LLM4Rec',
    desc: '写给多模态背景、没系统学过推荐的人:评估与偏差、长序列、排序 scaling、语义 ID、生成式推荐,以及搜索、图推荐、多任务、系统设计题与 Agent RL。',
    notes: [
      'recsys-llm4rec-guide',
      'recsys-eval-bias',
      'recsys-sequence-long',
      'recsys-scaling-ranking',
      'recsys-semantic-id',
      'recsys-llm4rec-paradigms',
      'recsys-multimodal-i2i',
      'recsys-ads-business',
      'recsys-multitask-scenario',
      'recsys-graph',
      'recsys-search',
      'recsys-system-design',
      'recsys-agent-rl',
      'recsys-interview-bank',
    ],
  },
  {
    key: 'wangshusen',
    name: '王树森推荐系统',
    desc: '工业推荐的经典链路:召回、排序、重排、冷启动与涨指标,每篇有手算例子,最后一篇是十六个算例的复习题。',
    notes: [
      'wangshusen-recommender-guide',
      'wangshusen-recommender-retrieval',
      'wangshusen-recommender-ranking',
      'wangshusen-recommender-reranking',
      'wangshusen-recommender-coldstart',
      'wangshusen-recommender-workbook',
    ],
  },
];

export const seriesOf = (id: string) => SERIES.find((s) => s.notes.includes(id));
