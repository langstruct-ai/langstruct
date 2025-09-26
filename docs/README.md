# LangStruct Documentation

This directory contains the source for the LangStruct documentation website, built with [Astro](https://astro.build) and [Starlight](https://starlight.astro.build).

## 🚀 Live Site

Visit the live documentation at [langstruct.dev](https://langstruct.dev)

## 📁 Project Structure

```text
docs/
├── src/
│   ├── components/          # Reusable UI components
│   ├── content/docs/        # Documentation content (MDX files)
│   ├── layouts/             # Page layouts
│   └── styles/              # Custom CSS styles
├── public/                  # Static assets
├── astro.config.mjs         # Astro configuration
└── package.json             # Dependencies and scripts
```

## 🧞 Commands

All commands are run from the `docs/` directory:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `pnpm install`            | Installs dependencies                            |
| `pnpm dev`                | Starts local dev server at `localhost:4321`     |
| `pnpm build`              | Build production site to `./dist/`              |
| `pnpm preview`            | Preview build locally, before deploying         |
| `pnpm astro ...`          | Run CLI commands like `astro add`, `astro check` |
| `pnpm astro -- --help`    | Get help using the Astro CLI                     |

## 🖊️ Editing Documentation

Documentation files are located in `src/content/docs/` and written in MDX (Markdown + JSX).

### Adding New Pages

1. Create a new `.mdx` file in `src/content/docs/`
2. Add frontmatter with title and description
3. The file path determines the URL (e.g., `getting-started.mdx` → `/getting-started/`)

### Updating Navigation

Navigation is automatically generated from the file structure. To customize:

1. Edit the sidebar configuration in `astro.config.mjs`
2. Use frontmatter to set custom titles or ordering

## 🎨 Components

The site includes custom components for enhanced documentation:

- `<CodeDemo>` - Interactive code examples
- `<Features>` - Feature highlight cards  
- `<Hero>` - Landing page hero section

## 🚀 Deployment

Documentation is automatically deployed via GitHub Actions when changes are pushed to the main branch.

**Manual deployment:**
```bash
cd docs
pnpm build
# Deploy dist/ directory to your hosting provider
```

## 📝 Writing Guidelines

- Use clear, concise language
- Include practical code examples
- Test all code examples before publishing
- Follow existing style and structure
- Add appropriate frontmatter to all pages

## 🔗 Links

- [Astro Documentation](https://docs.astro.build)
- [Starlight Documentation](https://starlight.astro.build)
- [MDX Documentation](https://mdxjs.com/docs/)

---

For questions about the documentation, please open an issue on GitHub.