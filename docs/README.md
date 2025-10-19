# CoreRec Documentation

This directory contains the complete documentation for CoreRec built with MkDocs Material.

## Setup

### Install Dependencies

```bash
# Install MkDocs and Material theme
pip install -r docs/requirements.txt

# Or install individually
pip install mkdocs mkdocs-material
```

### Preview Documentation Locally

```bash
# From the project root directory
mkdocs serve

# Then open http://127.0.0.1:8000 in your browser
```

### Build Documentation

```bash
# Build static HTML files
mkdocs build

# Output will be in site/ directory
```

## Deployment

### Deploy to GitHub Pages

```bash
# Deploy to gh-pages branch
mkdocs gh-deploy

# Your docs will be available at:
# https://vishesh9131.github.io/CoreRec/
```

### Manual Deployment

```bash
# Build the site
mkdocs build

# Deploy the site/ directory to your hosting service
```

## Documentation Structure

```
docs/
├── index.md                      # Homepage
├── getting-started/              # Installation & quick start
├── user-guide/                   # User guides
├── api/                          # API reference
├── engines/                      # Engine documentation
│   ├── unionized-filter/         # Collaborative filtering
│   ├── content-filter/           # Content-based filtering
│   └── deep-learning/            # Deep learning models
├── core/                         # Core components
│   └── towers/                   # Tower architectures
├── training/                     # Training guides
├── data/                         # Data handling
├── utilities/                    # Utility functions
├── examples/                     # Code examples
├── testing/                      # Testing guides
├── advanced/                     # Advanced topics
├── contributing/                 # Contributing guidelines
├── about/                        # About & contact
├── stylesheets/                  # Custom CSS
├── javascripts/                  # Custom JS
└── includes/                     # Reusable snippets
```

## Configuration

The site configuration is in `mkdocs.yml` at the project root.

### Key Features Enabled

- **Material Theme**: Modern, responsive design
- **Dark/Light Mode**: Toggle between themes
- **Search**: Full-text search
- **Navigation**: Tabbed navigation with sections
- **Code Highlighting**: Syntax highlighting for all languages
- **Math Support**: LaTeX/MathJax for equations
- **Mermaid Diagrams**: Flowcharts and diagrams
- **Admonitions**: Note, tip, warning boxes
- **Git Integration**: Last update dates on pages

## Writing Documentation

### Markdown Files

All documentation is written in Markdown (`.md` files).

### Example Page

```markdown
# Page Title

Introduction paragraph.

## Section

Content here.

### Code Example

\`\`\`python
from corerec.engines.dcn import DCN

model = DCN(embedding_dim=64)
model.fit(user_ids, item_ids, ratings)
\`\`\`

### Admonitions

!!! note
    This is a note.

!!! tip
    This is a tip.

!!! warning
    This is a warning.
```

### Adding New Pages

1. Create the markdown file in the appropriate directory
2. Add it to the `nav` section in `mkdocs.yml`
3. Preview locally with `mkdocs serve`

## Custom Styling

### CSS

Edit `docs/stylesheets/extra.css` for custom styles.

### JavaScript

Edit `docs/javascripts/mathjax.js` for custom JavaScript.

## Troubleshooting

### Build Errors

If you encounter build errors:

```bash
# Clean build
rm -rf site/
mkdocs build --clean
```

### Plugin Errors

If plugins fail to load:

```bash
# Reinstall dependencies
pip install --upgrade -r docs/requirements.txt
```

### Port Already in Use

If port 8000 is in use:

```bash
# Use a different port
mkdocs serve -a 127.0.0.1:8001
```

## Contributing

When contributing documentation:

1. Follow the existing structure
2. Use proper Markdown formatting
3. Add code examples where appropriate
4. Test locally before committing
5. Check for broken links

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## Support

For documentation issues:
- Open an issue on [GitHub](https://github.com/vishesh9131/CoreRec/issues)
- Contact: sciencely98@gmail.com


