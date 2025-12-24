# VectorReVamp Documentation

Welcome to the VectorReVamp documentation! This comprehensive documentation suite provides everything you need to understand, install, configure, and use VectorReVamp's intelligent test generation capabilities.

## 🚀 Quick Access

### Option 1: Python Script (Recommended)
```bash
# Open documentation in browser with local server
python open_docs.py

# Start server only (for multiple access)
python open_docs.py --server

# Open HTML file directly (no server)
python open_docs.py --file

# Use custom port
python open_docs.py --port 8080
```

### Option 2: Shell Script
```bash
# Open documentation in browser
./open_docs.sh

# Start server only
./open_docs.sh --server

# Open HTML file directly
./open_docs.sh --file

# Use custom port
./open_docs.sh --port 8080
```

### Option 3: Manual Access
If the scripts don't work, you can manually open:
- **File**: `docs/index.html` in your web browser
- **Server**: Run `cd docs && python -m http.server 8000` then visit `http://localhost:8000`

## 📚 Documentation Structure

### 📖 Getting Started
- **[Installation Guide](docs/pages/installation.html)** - Complete setup instructions
- **[Quick Start](docs/pages/installation.html#quick-start)** - Get running in 5 minutes
- **[Architecture Overview](docs/pages/architecture.html)** - How VectorReVamp works

### 🔧 Core Components
- **[API Reference](docs/pages/api.html)** - Complete API documentation
- **[HarnessRunner](docs/pages/api.html#harness-runner)** - Main orchestration class
- **[TemplateEngine](docs/pages/api.html#template-engine)** - Intelligent template management
- **[QualityValidator](docs/pages/api.html#quality-validator)** - Test quality assurance
- **[ParallelEngine](docs/pages/api.html#parallel-engine)** - Resource-aware parallel processing

### 🎯 Advanced Features
- **[Static Analysis Engine](docs/pages/architecture.html#analysis)** - Deep code analysis
- **[Template Evolution](docs/pages/architecture.html#evolution)** - Self-improving templates
- **[Domain Intelligence](docs/pages/architecture.html#domain)** - Context-aware generation
- **[Plugin System](docs/pages/architecture.html#plugins)** - Extensible architecture

### 🎮 Live Demo
- **[Interactive Demo](docs/pages/demonstration.html)** - Try VectorReVamp live
- **[Test Generation Demo](docs/pages/demonstration.html#testing-demo)** - See it in action

## 🎨 Documentation Features

### 🌙 Dark/Light Theme
- Toggle between dark and light themes using the moon/sun icon
- Theme preference is saved automatically

### 📱 Responsive Design
- Fully responsive design works on desktop, tablet, and mobile
- Optimized for all screen sizes

### 🔍 Search Functionality
- Built-in search through all documentation pages
- Quick navigation to relevant sections

### 📋 Code Highlighting
- Syntax highlighting for Python, Bash, JSON, and more
- Copy-to-clipboard functionality for code examples

### 🧭 Navigation
- Sidebar navigation for easy browsing
- Breadcrumb navigation
- Quick jump links between related sections

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: 4GB minimum, 8GB+ recommended
- **Storage**: 1GB free space minimum
- **OS**: Linux, macOS, or Windows with WSL

### Browser Support
- **Chrome/Edge**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Mobile browsers**: iOS Safari, Chrome Mobile

## 🚀 Quick Start Guide

1. **Open Documentation**
   ```bash
   python open_docs.py
   ```

2. **Follow Installation Guide**
   - Visit the [Installation](docs/pages/installation.html) page
   - Choose your installation method
   - Follow the step-by-step instructions

3. **Try the Demo**
   - Go to [Live Demo](docs/pages/demonstration.html)
   - Experiment with test generation
   - See VectorReVamp in action

4. **Read Architecture**
   - Check out [Architecture](docs/pages/architecture.html)
   - Understand how components work together
   - Learn about advanced features

## 🔧 Configuration

### Documentation Server
The documentation runs on a local HTTP server for the best experience. You can:

- **Change Port**: Use `--port` option to run on different port
- **Direct File Access**: Use `--file` option to open HTML directly
- **Server Only**: Use `--server` option for headless access

### Browser Settings
For the best experience:
- Enable JavaScript
- Allow local file access (for direct file opening)
- Use a modern browser with CSS Grid support

## 🆘 Troubleshooting

### Documentation Won't Load
```bash
# Check if files exist
ls -la docs/
ls -la docs/pages/

# Try different port
python open_docs.py --port 8080

# Open directly in browser
python open_docs.py --file
```

### Server Won't Start
```bash
# Check Python installation
python --version

# Check if port is available
netstat -an | grep 8000

# Try different port
PORT=8081 python open_docs.py
```

### Browser Issues
- **File not opening**: Try `--server` mode instead of `--file`
- **Styles not loading**: Check browser console for 404 errors
- **JavaScript errors**: Ensure JavaScript is enabled

### Permission Issues
```bash
# Make scripts executable
chmod +x open_docs.sh
chmod +x open_docs.py
```

## 📞 Support

### Getting Help
1. **Documentation**: Check the [troubleshooting](docs/pages/installation.html#troubleshooting) section
2. **GitHub Issues**: Report bugs or request features
3. **Community**: Join discussions and get help from the community

### Contributing
- **Report Issues**: Found a documentation bug? Let us know!
- **Suggest Improvements**: Have ideas for better docs? Share them!
- **Contribute**: Help improve the documentation

## 📊 Documentation Stats

- **Total Pages**: 6 comprehensive guides
- **Code Examples**: 50+ executable examples
- **API References**: Complete coverage of all classes and methods
- **Interactive Elements**: Theme toggle, search, navigation
- **Mobile Optimized**: Responsive design for all devices

## 🔄 Updates

The documentation is updated with each VectorReVamp release. To get the latest:

```bash
# Update VectorReVamp
pip install --upgrade vectorrevamp

# Re-open documentation
python open_docs.py
```

## 🎯 What's Included

### 📚 User Guides
- Installation and setup
- Configuration options
- Usage examples
- Best practices

### 🔧 API Documentation
- Complete class references
- Method signatures and parameters
- Usage examples
- Inheritance diagrams

### 🏗️ Architecture Docs
- System design and components
- Data flow diagrams
- Quality metrics
- Performance characteristics

### 🎮 Interactive Demo
- Live test generation examples
- Interactive configuration
- Real-time feedback

---

**Ready to explore VectorReVamp? Let's get started!**

```bash
python open_docs.py
```

*VectorReVamp - Intelligent Test Generation for the Modern Developer*
