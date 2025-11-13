# Release Checklist for LibFWHT

## Pre-Release Steps

### 1. Update Version Numbers
- [x] `python/pyfwht/_version.py` → 1.1.0
- [x] `python/pyproject.toml` → 1.1.0
- [x] `README.md` badge → 1.1.0
- [x] `include/fwht.h` → Update `FWHT_VERSION_*` macros (1.1.0)

### 2. Verify Documentation
- [x] README.md updated with latest benchmarks
- [x] Benchmark tables restructured for clarity
- [x] Performance claims updated
- [ ] Python README synced if needed

### 3. Testing
- [ ] Run full test suite: `make clean && make test`
- [ ] Test Python package: `cd python && python -m pytest tests/`
- [ ] Verify on GPU server if possible
- [ ] Check GitHub Actions CI passes

### 4. Clean Repository
- [ ] All changes committed
- [ ] No uncommitted files
- [ ] Git status clean

## Release Process

### Option 1: GitHub Release (Automated PyPI)
```bash
# 1. Push to GitHub
git push origin main

# 2. Create a new release on GitHub
# - Go to: https://github.com/hadipourh/fwht/releases/new
# - Tag: v1.1.0
# - Title: Version 1.1.0
# - Description: See CHANGELOG below
# - Publish release

# 3. GitHub Actions will automatically:
#    - Build the Python package
#    - Publish to PyPI (if configured with OIDC trusted publisher)
```

### Option 2: Manual PyPI Release
```bash
# 1. Build the package
cd python
python -m build

# 2. Check the build
twine check dist/*

# 3. Test on TestPyPI first
twine upload --repository testpypi dist/*

# 4. Test installation
pip install --index-url https://test.pypi.org/simple/ pyfwht

# 5. Upload to PyPI
twine upload dist/*
```

### Option 3: Workflow Dispatch (Manual Trigger)
```bash
# 1. Push changes
git push origin main

# 2. Go to GitHub Actions
# - Navigate to "Publish to PyPI" workflow
# - Click "Run workflow"
# - Choose TestPyPI or PyPI
```

## CHANGELOG for v1.1.0

### Highlights
- Restructured benchmark documentation for clarity
- Separated CPU (macOS M4) and GPU (Linux A30) results
- Removed misleading cross-platform performance comparisons
- Updated version to 1.1.0 across all components

### Documentation Improvements
- CPU benchmarks now clearly separated by platform
- GPU results presented independently without CPU comparison
- Performance insights refined to avoid platform-specific claims
- Clearer guidance on when to use CPU vs GPU backends

### Technical Changes
- Version bumped to 1.1.0 in all components
- No functional changes to library code
- Improved reproducibility documentation for benchmarks

## Post-Release

### 1. Verify PyPI Publication
- [ ] Check https://pypi.org/project/pyfwht/
- [ ] Test installation: `pip install pyfwht==1.1.0`
- [ ] Verify package metadata

### 2. Update Documentation
- [ ] Update package documentation if needed
- [ ] Announce release (optional)

### 3. Tag Repository
```bash
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0
```

## PyPI Trusted Publisher Setup (One-time)

To enable automatic PyPI publishing from GitHub Actions:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - PyPI Project Name: `pyfwht`
   - Owner: `hadipourh`
   - Repository name: `fwht`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Save the publisher

This eliminates the need for API tokens and enables secure automated releases.
