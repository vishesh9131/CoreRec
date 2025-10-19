# CoreRec Documentation Guide

Complete Material for MkDocs documentation has been created for CoreRec!

## 📚 What's Been Created

### 1. Complete Documentation Structure

```
CoreRec/
├── mkdocs.yml                    # Main configuration file
├── docs/                         # Documentation source
│   ├── index.md                  # Homepage
│   ├── getting-started/          # Installation, quickstart, architecture
│   ├── user-guide/               # Comprehensive user guides
│   ├── api/                      # API reference (BaseRecommender, interfaces)
│   ├── engines/                  # All three engines documented
│   │   ├── unionized-filter/     # Collaborative filtering
│   │   ├── content-filter/       # Content-based filtering
│   │   └── deep-learning/        # Deep learning models
│   ├── core/                     # Core components (towers, encoders, losses)
│   ├── training/                 # Training pipeline and optimization
│   ├── data/                     # Data handling and preprocessing
│   ├── utilities/                # Utilities and helper functions
│   ├── examples/                 # Real-world examples
│   ├── testing/                  # Testing guide
│   ├── advanced/                 # Advanced topics
│   ├── contributing/             # Contributing guidelines
│   └── about/                    # About, license, contact
└── BUILD_DOCS.md                 # This guide
```

### 2. Key Features

✅ **100+ Pages of Documentation**
✅ **Material Theme** with dark/light mode
✅ **Comprehensive API Reference**
✅ **Real-World Examples**
✅ **Testing Guides**
✅ **Interactive Navigation**
✅ **Search Functionality**
✅ **Code Syntax Highlighting**
✅ **Mermaid Diagrams**
✅ **Math Support (MathJax)**

## 🚀 Quick Start

### Install MkDocs

```bash
# Install all dependencies
pip install -r docs/requirements.txt
```

### Preview Documentation

```bash
# Start development server
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Build Documentation

```bash
# Build static HTML
mkdocs build

# Output in site/ directory
```

## 🌐 Deployment Options

### Option 1: GitHub Pages (Recommended)

```bash
# One command deployment
mkdocs gh-deploy

# Docs will be live at:
# https://vishesh9131.github.io/CoreRec/
```

### Option 2: Read the Docs

1. Connect your GitHub repo to [readthedocs.org](https://readthedocs.org)
2. Configuration is already set in `mkdocs.yml`
3. Auto-deploys on each commit

### Option 3: Netlify

1. Drag and drop the `site/` folder to [Netlify](https://netlify.com)
2. Or connect your GitHub repo for auto-deployment

### Option 4: Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

## 📖 Documentation Highlights

### Getting Started
- **Installation Guide**: Step-by-step installation for all platforms
- **Quick Start**: 5-minute tutorial to build first recommender
- **Architecture Overview**: Complete system architecture

### Engines Documentation
- **Unionized Filter Engine**: 50+ collaborative filtering algorithms
- **Content Filter Engine**: 40+ content-based methods
- **Deep Learning Models**: 6 state-of-the-art models (DCN, DeepFM, GNNRec, MIND, NASRec, SASRec)

### API Reference
- **BaseRecommender**: Complete API documentation
- **All Methods**: fit(), predict(), recommend(), save(), load()
- **Type Hints**: Full type annotations
- **Code Examples**: For every method

### Examples
- **Quick Start Examples**: For each engine
- **Engine-Specific Examples**: DCN, DeepFM, GNNRec, MIND, NASRec, SASRec
- **Unionized Filter Examples**: Fast, SAR, RBM, RLRMC, GeoMLC
- **Advanced Examples**: Instagram Reels, YouTube MoE, DIEN
- **Complete Workflows**: End-to-end examples

### Core Components
- **Towers**: MLP, CNN, Transformer, Fusion towers
- **Encoders**: Feature encoding and transformation
- **Embedding Tables**: Efficient embedding management
- **Losses**: Multiple loss functions

### Testing
- **Unit Tests**: How to write and run unit tests
- **Integration Tests**: Testing complete workflows
- **Smoke Tests**: Quick sanity checks
- **Running Tests**: Complete test guide

## 🎨 Customization

### Change Theme Colors

Edit `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: indigo  # Change to: blue, red, green, etc.
    accent: indigo
```

### Add Custom CSS

Edit `docs/stylesheets/extra.css`:

```css
:root {
  --md-primary-fg-color: #your-color;
}
```

### Update Navigation

Edit `mkdocs.yml` under `nav:` section to add/remove pages.

## 📝 Adding New Content

### Create a New Page

1. Create markdown file: `docs/section/new-page.md`
2. Add to navigation in `mkdocs.yml`:

```yaml
nav:
  - Section:
    - New Page: section/new-page.md
```

3. Preview: `mkdocs serve`
4. Commit and push

### Example Page Template

```markdown
# Page Title

Brief introduction.

## Section 1

Content here.

### Code Example

\`\`\`python
from corerec.engines.dcn import DCN

model = DCN(embedding_dim=64)
\`\`\`

### Notes

!!! note
    Important information.

!!! tip
    Helpful tip.

## See Also

- [Related Page](other-page.md)
- [API Reference](../api/index.md)
```

## 🔧 Maintenance

### Update Dependencies

```bash
pip install --upgrade -r docs/requirements.txt
```

### Check for Broken Links

```bash
# Install linkchecker
pip install linkchecker

# Start server
mkdocs serve &

# Check links
linkchecker http://127.0.0.1:8000
```

### Clean Build

```bash
# Remove old build
rm -rf site/

# Clean build
mkdocs build --clean
```

## 🌟 Key Documentation Pages

### Must-Read Pages

1. **[Homepage](docs/index.md)** - Overview and features
2. **[Installation](docs/getting-started/installation.md)** - Setup guide
3. **[Quick Start](docs/getting-started/quickstart.md)** - 5-minute tutorial
4. **[Architecture](docs/getting-started/architecture.md)** - System design
5. **[API Reference](docs/api/base-recommender.md)** - BaseRecommender API
6. **[Engines Overview](docs/engines/index.md)** - All engines
7. **[Examples](docs/examples/index.md)** - Code examples
8. **[Testing](docs/testing/index.md)** - Testing guide

### Engine Documentation

- **[Unionized Filter Engine](docs/engines/unionized-filter/index.md)** - Collaborative filtering
- **[Content Filter Engine](docs/engines/content-filter/index.md)** - Content-based filtering
- **[Deep Learning Models](docs/engines/deep-learning/index.md)** - SOTA models

### Component Documentation

- **[Towers](docs/core/towers/index.md)** - Neural network towers
- **[Core Components](docs/core/index.md)** - Building blocks
- **[Utilities](docs/utilities/index.md)** - Helper functions

## 📊 Analytics

To enable Google Analytics, add to `mkdocs.yml`:

```yaml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX
```

## 🤝 Contributing

### Documentation Contributions

1. Fork the repository
2. Create a branch: `git checkout -b docs/my-improvement`
3. Make changes in `docs/` directory
4. Test locally: `mkdocs serve`
5. Commit and push
6. Create pull request

### Style Guidelines

- Use clear, concise language
- Include code examples
- Add cross-references
- Test all code examples
- Use proper Markdown formatting

## 📱 Mobile-Friendly

The documentation is fully responsive and works great on:
- Desktop browsers
- Tablets
- Mobile phones

## 🔍 Search

The documentation includes full-text search:
- Press `/` to focus search
- Type to search across all pages
- Navigate results with arrow keys

## ⌨️ Keyboard Shortcuts

- `/` - Focus search
- `n` - Next page
- `p` - Previous page
- `s` - Toggle sidebar

## 🎯 Next Steps

1. **Preview the Documentation**
   ```bash
   mkdocs serve
   ```

2. **Deploy to GitHub Pages**
   ```bash
   mkdocs gh-deploy
   ```

3. **Share with Team**
   - Send documentation URL
   - Encourage contributions
   - Gather feedback

4. **Keep Updated**
   - Update docs with code changes
   - Add new examples
   - Improve based on user feedback

## 📞 Support

Need help with documentation?

- **GitHub Issues**: https://github.com/vishesh9131/CoreRec/issues
- **Email**: sciencely98@gmail.com
- **MkDocs Docs**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/

## 🎉 Congratulations!

You now have a complete, professional documentation site for CoreRec!

**Documentation includes:**
- ✅ Installation guides
- ✅ Quick start tutorials
- ✅ Complete API reference
- ✅ 100+ algorithm documentation
- ✅ Real-world examples
- ✅ Testing guides
- ✅ Best practices
- ✅ Contributing guidelines

**Ready to share with the world!** 🚀
