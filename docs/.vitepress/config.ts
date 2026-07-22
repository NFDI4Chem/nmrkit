import { defineConfigWithTheme } from "vitepress";
import type { ThemeConfig } from '@hempworks/pilgrim'
import config from '@hempworks/pilgrim/config'

export default defineConfigWithTheme<ThemeConfig>({
    extends: config,
    title: 'NMRKit',
    description: 'A collection of powerful microservices designed to simplify your NMR data processing and analysis',
    base: '/nmrkit/',
    cleanUrls: false,
    srcDir: 'src',
    themeConfig: {
        logo: {
            light: '/logo.svg',
            dark: '/logo-dark.svg',
        },
        nav: [
            { text: 'Home', link: '/nmrkit/introduction' },
            { text: 'API Reference', link: '/getting-started/api-reference' },
            { text: 'Scalar (dev)', link: 'https://dev.nmrkit.nmrxiv.org/latest/docs' },
            { text: 'GitHub', link: 'https://github.com/NFDI4Chem/nmrkit' }
        ],
        sidebar: [
            {
              text: 'Getting Started',
              items: [
                { text: 'Introduction', link: '/introduction' },
                { text: 'API Reference', link: '/getting-started/api-reference' },
                { text: 'Architecture', link: '/getting-started/architecture' },
                { text: 'Versions', link: '/getting-started/versions' },
              ]
            },
            {
              text: 'Deployment',
              items: [
                { text: 'Overview', link: '/deployment/overview' },
                { text: 'Docker', link: '/deployment/docker' },
                { text: 'Cluster Deployment (K8S)', link: '/deployment/cluster-deployment' }
              ]
            },
            {
              text: 'Services',
              items: [
                { text: 'nmr-cli', link: '/services/nmr-cli' },
                { text: 'nmr-respredict', link: '/services/nmr-respredict' },
              ]
            },
            {
              text: 'Modules',
              items: [
                { text: 'Chemistry', link: '/modules/chemistry' },
                { text: 'Spectra', link: '/modules/spectra' },
                { text: 'Converter', link: '/modules/converter' },
                { text: 'Prediction', link: '/modules/prediction' },
                { text: 'Registration', link: '/modules/registration' },
                { text: 'Search (planned)', link: '/modules/search' },
                { text: 'Training (planned)', link: '/modules/training' },
                { text: 'Validation (planned)', link: '/modules/validation' },
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