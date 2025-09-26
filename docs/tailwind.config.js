/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'Monaco', 'monospace'],
      },
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        gray: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        success: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        warning: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        }
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            color: 'rgb(var(--tw-prose-body))',
            '--tw-prose-body': '64 74 91',
            '--tw-prose-headings': '30 41 59',
            '--tw-prose-lead': '64 74 91',
            '--tw-prose-links': '14 165 233',
            '--tw-prose-bold': '30 41 59',
            '--tw-prose-counters': '100 116 139',
            '--tw-prose-bullets': '203 213 225',
            '--tw-prose-hr': '226 232 240',
            '--tw-prose-quotes': '30 41 59',
            '--tw-prose-quote-borders': '226 232 240',
            '--tw-prose-captions': '100 116 139',
            '--tw-prose-code': '30 41 59',
            '--tw-prose-pre-code': '226 232 240',
            '--tw-prose-pre-bg': '15 23 42',
            '--tw-prose-th-borders': '203 213 225',
            '--tw-prose-td-borders': '226 232 240',
          },
        },
        dark: {
          css: {
            '--tw-prose-body': '148 163 184',
            '--tw-prose-headings': '241 245 249',
            '--tw-prose-lead': '148 163 184',
            '--tw-prose-links': '56 189 248',
            '--tw-prose-bold': '241 245 249',
            '--tw-prose-counters': '100 116 139',
            '--tw-prose-bullets': '71 85 105',
            '--tw-prose-hr': '30 41 59',
            '--tw-prose-quotes': '241 245 249',
            '--tw-prose-quote-borders': '30 41 59',
            '--tw-prose-captions': '100 116 139',
            '--tw-prose-code': '241 245 249',
            '--tw-prose-pre-code': '226 232 240',
            '--tw-prose-pre-bg': '15 23 42',
            '--tw-prose-th-borders': '71 85 105',
            '--tw-prose-td-borders': '30 41 59',
          },
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-up': 'scaleUp 0.2s ease-out',
        'gradient': 'gradient 15s ease infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleUp: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        }
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}