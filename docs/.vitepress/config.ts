import { defineConfigWithTheme } from "vitepress";
import type { ThemeConfig } from '@hempworks/pilgrim'
import config from '@hempworks/pilgrim/config'

export default defineConfigWithTheme<ThemeConfig>({
    extends: config,
    title: 'Envoyer',
    description: 'Zero Downtime PHP Deployments',
    base: '/nmrkit/',
    cleanUrls: false,
    srcDir: 'src',

    themeConfig: {
        logo: {
            light: '/logo.png',
            dark: '/logo-dark.png',
        },
        nav: [
            { text: 'Home', link: '/introduction' },
            { text: 'API', link: '' },
        ],
        sidebar: [
            {
              text: 'Getting Started',
              items: [
                { text: 'Introduction', link: '/introduction' },
                { text: 'Versions', link: '/getting-started/versions' },
                { text: 'Architecture', link: '/getting-started/architecture' },
              ]
            },
            {
              text: 'Deployment',
              items: [
                //{ text: 'Public API', link: '/public-api' },
                { text: 'Docker', link: '/deployment/docker' },
                { text: 'Cluster Deployment (K8S)', link: '/deployment/cluster-deployment' }
              ]
            },
            {
              text: 'Modules',
              items: [
                { text: 'Registration', link: '/modules/registration' },
                { text: 'Search', link: '/modules/search' },
                { text: 'Prediction', link: '/modules/prediction' },
                { text: 'Spectra', link: '/modules/spectra' },
                { text: 'Training', link: '/modules/training' },
                { text: 'Validation', link: 'modules/validation' },
              ]
            },
            {
              text: 'Development',
              items: [
                { text: 'Local Installation', link: '/development/installation' },
                { text: 'License', link: '/development/license' },
                { text: 'Issues/Feature requests', link: '/development/issues' },
                { text: 'Contributors', link: '/development/contributors' }
              ]
            }
          ],
    },
    vite: {
        server: {
            host: true,
            fs: {
                // for when developing with locally linked theme
                allow: ['../..']
            }
        },
    }
})