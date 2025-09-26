// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import catppuccin from '@catppuccin/starlight';

// https://astro.build/config
export default defineConfig({
  site: 'https://langstruct.dev',
  integrations: [
    starlight({
      title: 'LangStruct',
      description: 'LLM-powered structured information extraction using DSPy optimization',
      logo: {
        src: './src/assets/logo.svg',
        replacesTitle: true,
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/langstruct-ai/langstruct',
        },
        {
          icon: 'discord',
          label: 'Discord',
          href: 'https://discord.gg/langstruct',
        },
        {
          icon: 'twitter',
          label: 'Twitter',
          href: 'https://twitter.com/langstruct',
        },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Quick Start', link: '/quickstart/' },
            { label: 'Installation', link: '/installation/' },
            { label: 'Introduction', link: '/getting-started/' },
            { label: 'Why LangStruct?', link: '/why-langstruct/' },
            { label: 'Why We Chose DSPy', link: '/why-dspy/' },
          ],
        },
        {
          label: 'Key Features',
          items: [
            { label: 'Query Parsing', link: '/query-parsing/' },
            { label: 'Source Grounding', link: '/source-grounding/' },
            { label: 'Auto-Optimization', link: '/optimization/' },
            { label: 'RAG Integration', link: '/rag-integration/' },
            { label: 'Refinement', link: '/refinement/' },
            { label: 'Save & Load Extractors', link: '/persistence/' },
          ],
        },
        {
          label: 'Examples',
          items: [
            { label: 'Progressive Examples', link: '/examples/' },
            { label: 'Financial Documents', link: '/examples/financial-documents/' },
            { label: 'Medical Records', link: '/examples/medical-records/' },
          ],
        },
      ],
      customCss: [
        './src/styles/custom.css',
      ],
      components: {
        ThemeProvider: './src/components/ThemeToggle.astro',
      },
      plugins: [
        catppuccin({
          // Use Latte for light mode, Mocha for dark mode
        }),
      ],
    }),
  ],
});
