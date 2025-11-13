# Release Checklist for LibFWHT

## Pre-Release Steps

### 1. Update Version Numbers
- [x] `python/pyfwht/_version.py` → 1.0.1
- [x] `README.md` badge → 1.0.1
- [ ] `include/fwht.h` → Update `FWHT_VERSION_*` macros if needed

### 2. Verify Documentation
- [x] README.md updated with latest benchmarks
- [x] Key features clearly documented
- [x] Performance insights documented
- [ ] Python README synced if needed

### 3. Testing
- [ ] Run full test suite: `make clean && make test`
- [ ] Test Python package: `cd python && python -m pytest tests/`
- [ ] Verify on GPU server if possible
- [ ] Check GitHub Actions CI passes

### 4. Clean Repository
- [x] All changes committed
- [ ] No uncommitted files
- [ ] Git status clean

## Release Process

### Option 1: GitHub Release (Automated PyPI)
```bash
# 1. Push to GitHub
git push origin main

# 2. Create a new release on GitHub
# - Go to: https://github.com/hadipourh/fwht/releases/new
# - Tag: v1.0.1
# - Title: Version 1.0.1
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

## CHANGELOG for v1.0.1

### Highlights
- Enhanced README with memory-bandwidth analysis
- Updated benchmarks with A30 datacenter GPU results
- Clarified performance characteristics and architecture recommendations
- Python package synced with C library

### Performance Insights
- Documented memory-bound nature of FWHT algorithm
- A30 GPU achieves 2.7-3.5× speedup with excellent consistency
- OpenMP multi-threading: 2.7× speedup on 10 cores

### Documentation
- Cleaner feature-focused README
- Removed version-specific implementation details
- Added GPU architecture comparison (HBM vs GDDR6X)

## Post-Release

### 1. Verify PyPI Publication
- [ ] Check https://pypi.org/project/pyfwht/
- [ ] Test installation: `pip install pyfwht==1.0.1`
- [ ] Verify package metadata

### 2. Update Documentation
- [ ] Update package documentation if needed
- [ ] Announce release (optional)

### 3. Tag Repository
```bash
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
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
