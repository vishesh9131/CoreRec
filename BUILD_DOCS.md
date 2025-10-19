# Building CoreRec Documentation

Complete guide to building and deploying CoreRec documentation.

## Quick Start

### Install Dependencies

```bash
pip install -r docs/requirements.txt
```

### Preview Locally

```bash
# Start development server
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Build Static Site

```bash
# Build documentation
mkdocs build

# Output in site/ directory
```

## Deployment

### Option 1: GitHub Pages (Recommended)

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy

# Site will be available at:
# https://vishesh9131.github.io/CoreRec/
```

### Option 2: Manual Deployment

```bash
# Build the site
mkdocs build

# Deploy site/ directory to your hosting service
# Examples:
# - Netlify: drag and drop site/ folder
# - Vercel: connect GitHub repo
# - AWS S3: upload site/ contents
```

### Option 3: Read the Docs

1. Go to [readthedocs.org](https://readthedocs.org/)
2. Import CoreRec repository
3. Configure using `mkdocs.yml`
4. Builds automatically on push

## Documentation Structure

```
CoreRec/
├── mkdocs.yml              # Configuration file
├── docs/                   # Documentation source
│   ├── index.md            # Homepage
│   ├── getting-started/    # Getting started guides
│   ├── user-guide/         # User guides
│   ├── api/                # API reference
│   ├── engines/            # Engine documentation
│   ├── core/               # Core components
│   ├── examples/           # Examples
│   └── ...                 # Other sections
└── site/                   # Built site (generated)
```

## Features

### Material Theme

Modern, responsive documentation theme with:

- Dark/light mode toggle
- Mobile-friendly navigation
- Search functionality
- Code syntax highlighting
- Table of contents
- Keyboard navigation

### Enhanced Markdown

- **Admonitions**: Note, tip, warning, etc.
- **Code blocks**: With syntax highlighting
- **Tables**: Markdown tables with sorting
- **Mermaid**: Diagrams and flowcharts
- **Math**: LaTeX/MathJax equations
- **Tabs**: Tabbed content sections
- **Icons**: Material Design icons

### Navigation

- **Tabs**: Top-level navigation tabs
- **Sections**: Organized into logical sections
- **Search**: Full-text search across all pages
- **Breadcrumbs**: Path to current page
- **Table of contents**: Per-page TOC

## Customization

### Theme Colors

Edit in `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: indigo
    accent: indigo
```

### Custom CSS

Add styles in `docs/stylesheets/extra.css`:

```css
:root {
  --md-primary-fg-color: #3f51b5;
}
```

### Logo and Icons

```yaml
theme:
  logo: assets/logo.png
  favicon: assets/favicon.ico
```

## Content Guidelines

### File Naming

- Use lowercase with hyphens: `getting-started.md`
- Place in appropriate directory
- Keep URLs short and meaningful

### Markdown Style

- Use ATX-style headers (`#` not `===`)
- One sentence per line (for git diffs)
- Empty line before and after code blocks
- Use fenced code blocks with language

### Code Examples

Always include language identifier:

````markdown
```python
from corerec.engines.dcn import DCN

model = DCN(embedding_dim=64)
```
````

### Cross-References

Use relative links:

```markdown
See [Installation](../getting-started/installation.md) for details.
```

### Admonitions

Use for important information:

```markdown
!!! note
    This is important information.

!!! tip
    This is a helpful tip.

!!! warning
    This is a warning.
```

## Maintenance

### Check Links

```bash
# Install linkchecker
pip install linkchecker

# Check all links
linkchecker http://127.0.0.1:8000
```

### Update Dependencies

```bash
pip install --upgrade -r docs/requirements.txt
```

### Review Analytics

If using Google Analytics:

- Check `mkdocs.yml` for analytics configuration
- Monitor page views and popular sections
- Update based on user behavior

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.x
      - run: pip install -r docs/requirements.txt
      - run: mkdocs gh-deploy --force
```

## Troubleshooting

### Common Issues

**1. Plugin not found**

```bash
pip install mkdocs-material-extensions
```

**2. Build fails**

```bash
mkdocs build --clean --strict
```

**3. Search not working**

```bash
# Rebuild with clean cache
rm -rf site/
mkdocs build
```

**4. Math not rendering**

- Check MathJax configuration in `docs/javascripts/mathjax.js`
- Verify `pymdownx.arithmatex` is enabled

## Best Practices

1. **Preview before deploying**: Always run `mkdocs serve` to check
2. **Test all links**: Ensure no broken links
3. **Optimize images**: Compress images before adding
4. **Write clear examples**: Include complete, runnable code
5. **Update regularly**: Keep documentation in sync with code
6. **Version documentation**: Tag docs for each release

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
- [Mermaid Diagrams](https://mermaid-js.github.io/)

## Need Help?

- GitHub Issues: https://github.com/vishesh9131/CoreRec/issues
- Email: sciencely98@gmail.com
