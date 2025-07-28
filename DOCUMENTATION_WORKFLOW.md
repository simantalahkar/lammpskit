# Documentation Workflow Summary

## Dual Documentation Hosting Strategy

LAMMPSKit follows industry best practices with **dual documentation hosting**:

### 🎯 **Primary Documentation Sources:**
1. **Read the Docs**: `https://lammpskit.readthedocs.io/` (Professional, PyPI-integrated)
2. **GitHub Pages**: `https://simantalahkar.github.io/lammpskit/` (Developer-focused)

### ✅ **Why Both?**
- ✅ **Professional Standard**: Major packages (NumPy, Pandas, SciPy) use dual hosting
- ✅ **Redundancy**: If one service is down, documentation remains accessible
- ✅ **Different Audiences**: PyPI users → Read the Docs, GitHub users → Pages
- ✅ **SEO Benefits**: Multiple URLs increase discoverability
- ✅ **Feature Diversity**: Read the Docs (versioning, PDF) + GitHub Pages (GitHub integration)

## Single Unified Workflow: `docs.yml`

### 🎯 **Triggers:**
- **Branches**: `main`, `develop`
- **Events**: Push, Pull Request

### ✅ **Features (All Builds):**
1. **Build Documentation**: `sphinx -b html` with warnings as errors (`-W`)
2. **Link Checking**: `sphinx -b linkcheck` to verify all links work
3. **Upload Artifacts**: Documentation HTML available for download (30 days)
4. **Verification**: Tests that LAMMPSKit and Sphinx are properly installed

### 🚀 **GitHub Pages Deployment (Main Branch Only):**
- **Conditional**: Only runs on `main` branch pushes (not PRs or develop)
- **Setup Pages**: Configures GitHub Pages environment
- **Deploy**: Publishes documentation to GitHub Pages

### 🔄 **Workflow Logic:**
```
All builds:
  ├── Install dependencies (with fallback)
  ├── Build docs (-W for warnings as errors)
  ├── Check links
  └── Upload artifacts

Main branch pushes only:
  ├── Setup GitHub Pages
  ├── Upload Pages artifact
  └── Deploy to Pages
```

### 📊 **Benefits:**
- ✅ **Single workflow** (no redundancy)
- ✅ **Comprehensive testing** on all branches
- ✅ **Automatic deployment** to GitHub Pages
- ✅ **Artifact downloads** for manual inspection
- ✅ **Link validation** prevents broken documentation
- ✅ **Resource efficient** (no duplicate runs)

### 🔗 **Documentation Availability:**
1. **Read the Docs**: `https://lammpskit.readthedocs.io/` (Primary - Professional)
2. **GitHub Pages**: `https://simantalahkar.github.io/lammpskit/` (Secondary - Developer)
3. **Artifacts**: Download from GitHub Actions runs

### 📊 **Hosting Strategy Benefits:**
- 🎯 **Read the Docs**: Professional URL, PyPI integration, versioning, PDF export
- 🚀 **GitHub Pages**: GitHub integration, developer accessibility, project visibility
- 📦 **Artifacts**: Manual inspection, debugging, offline access

This dual hosting approach provides maximum accessibility while following industry standards for professional Python packages.
