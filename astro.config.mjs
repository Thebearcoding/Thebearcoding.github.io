import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://athebear.me',
  integrations: [sitemap()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'one-dark-pro',
      // 代码块写 ```python title="run_command.py" 时,把文件名带到 <pre data-title> 上,由页面脚本做成标题栏
      transformers: [{
        name: 'meta-title',
        pre(node) {
          const m = this.options.meta?.__raw?.match(/title="([^"]+)"/);
          if (m) node.properties['data-title'] = m[1];
        },
      }],
    },
  },
});
