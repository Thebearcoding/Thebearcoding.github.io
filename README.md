# athebear.me

个人主页与博客:研究笔记、想法、互动课程入口。

- 框架:[Astro](https://astro.build) · 设计:暖色 Anthropic 风(与课程视觉同源)
- 写笔记:在 `src/content/notes/` 新建 `.md`(frontmatter: title / date / tags / summary),push 后自动发布
- 本地预览:`npm install && npm run dev`
- 部署:GitHub Actions 自动构建发布到 GitHub Pages(见 `.github/workflows/deploy.yml`)

栏目:主页(`/`)· 笔记(`/notes/`)· 教学与学习(`/teaching/`)· RSS(`/rss.xml`)
