import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "NMR Predict",
  description: "Microservice designed to predict the NMR spectrum of a molecule quickly and accurately.",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Docs', link: '/introduction' },
      { text: 'API', link: '/introduction' },
    ],

    sidebar: [
      {
        text: 'Welcome',
        items: [
          { text: 'Introduction', link: '/introduction' },
          { text: 'Versions', link: '/versions' },
          { text: 'Architecture', link: '/architecture' },
        ]
      },
      {
        text: 'Usage',
        items: [
          //{ text: 'Public API', link: '/public-api' },
          { text: 'Docker', link: '/docker' },
          { text: 'Cluster Deployment (K8S)', link: '/cluster-deployment' }
        ]
      },
      {
        text: 'Modules',
        items: [
          { text: 'Lightweight-Registration', link: '/lightweight-registration' },
          { text: 'Spectra', link: '/spectra' },
          { text: 'Training', link: '/training' },
          { text: 'Validation', link: '/validation' },
        ]
      },
      {
        text: 'Development',
        items: [
          { text: 'Local Installation', link: '/installation' },
          { text: 'License', link: '/license' },
          { text: 'Issues/Feature requests', link: '/issues' },
          { text: 'Contributors', link: '/contributors' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
