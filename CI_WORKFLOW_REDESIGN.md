# CI/CD Workflow Redesign

## 🎯 Overview

Redesigned CI/CD pipeline with proper job dependencies to ensure quality gates pass before version bumping and deployment. This prevents broken versions from being created when tests fail.

## ❌ Previous Issues

### Problem: Race Conditions & No Dependencies
```
Push to main
    ↓
    ├─→ ci.yml (tests)        [independent]
    ├─→ commitizen.yml (bump) [independent] ⚠️ RUNS IMMEDIATELY
    └─→ deploy-docs.yml       [independent]

Result: Version could bump even if tests fail!
```

### Critical Issues:
1. ❌ Commitizen bumps version **before** CI completes
2. ❌ Docs could deploy with failing tests
3. ❌ No coordination between workflows
4. ❌ Could create broken releases

## ✅ New Design

### Single Unified Workflow: `main.yml`

```
Push to main
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 1: Quality Gates (Parallel)          │
│  ├─ Lint ✓                                 │
│  ├─ Type Check ✓                           │
│  ├─ Tests (Ubuntu + macOS) ✓               │
│  └─ Complexity Check ⚠️ (advisory)         │
└─────────────────────────────────────────────┘
                    ↓ ALL MUST PASS
┌─────────────────────────────────────────────┐
│ PHASE 2: Documentation Build               │
│  └─ Build MkDocs ✓                         │
└─────────────────────────────────────────────┘
                    ↓ MUST PASS
┌─────────────────────────────────────────────┐
│ PHASE 3: Deploy Documentation              │
│  └─ Deploy to GitHub Pages ✓               │
└─────────────────────────────────────────────┘
                    ↓ MUST PASS
┌─────────────────────────────────────────────┐
│ PHASE 4: Version Bump                      │
│  └─ Commitizen bump → creates tag v1.2.3   │
└─────────────────────────────────────────────┘
                    ↓ TAG CREATED
┌─────────────────────────────────────────────┐
│ PHASE 5: Create GitHub Release             │
│  └─ Release with auto-generated notes ✓    │
└─────────────────────────────────────────────┘
                    ↓ TAG v1.2.3 EXISTS
┌─────────────────────────────────────────────┐
│ release.yml: Publish to PyPI (separate)    │
│  └─ Build & publish package ✓               │
└─────────────────────────────────────────────┘
```

## 📋 Job Dependencies

```yaml
lint:           # No dependencies
typecheck:      # No dependencies
test:           # No dependencies
complexity:     # No dependencies (advisory)

build-docs:     needs: [lint, typecheck, test]
deploy-docs:    needs: [build-docs]
version-bump:   needs: [lint, typecheck, test, deploy-docs]
create-release: needs: [version-bump]

# Separate file:
release.yml:    triggered by: tags (v*)
```

## 🎯 Execution Logic

### For Pull Requests:
```
✅ Run: lint, typecheck, test, complexity
✅ Run: build-docs (verify docs compile)
❌ Skip: deploy-docs (not main branch)
❌ Skip: version-bump (not main branch)
❌ Skip: create-release (no version bump)
❌ Skip: release.yml (no tag)
```

### For Push to Main (All Pass):
```
✅ Phase 1: lint, typecheck, test → ALL PASS
✅ Phase 2: build-docs → PASS
✅ Phase 3: deploy-docs → PASS
✅ Phase 4: version-bump → creates tag
✅ Phase 5: create-release → creates GitHub release
✅ Trigger: release.yml → publishes to PyPI
```

### For Push to Main (Tests Fail):
```
❌ Phase 1: lint ✓, typecheck ✓, test ❌ FAIL
🛑 STOP - Pipeline halts

Result:
- No docs deployment
- No version bump
- No release
- No PyPI publish
```

### For Push to Main (Docs Build Fails):
```
✅ Phase 1: lint ✓, typecheck ✓, test ✓
❌ Phase 2: build-docs ❌ FAIL (broken docs)
🛑 STOP - Pipeline halts

Result:
- No docs deployment
- No version bump
- No release
```

## 📊 Workflow Files

### Keep (Modified):
- ✅ `.github/workflows/main.yml` (NEW - unified CI/CD)
- ✅ `.github/workflows/release.yml` (KEEP - PyPI publish on tags)
- ✅ `.github/workflows/codeql.yml` (KEEP - security scanning)

### Delete (Replaced by main.yml):
- ❌ `.github/workflows/ci.yml` (merged into main.yml)
- ❌ `.github/workflows/commitizen.yml` (merged into main.yml)
- ❌ `.github/workflows/deploy-docs.yml` (merged into main.yml)

## ✨ Benefits

### 1. **Safety First**
- Version bumps **only after** all checks pass
- No broken versions created
- No deploying broken docs

### 2. **Clear Dependencies**
- Visual dependency graph in GitHub UI
- Easy to understand execution flow
- Predictable behavior

### 3. **Fast Feedback**
- PRs run quality checks only
- Main branch gets full pipeline
- Parallel execution where possible

### 4. **Resource Efficient**
- Jobs run in parallel when possible
- Skip unnecessary steps on PRs
- Cancel in-progress runs on new pushes

### 5. **Better Observability**
- Single workflow to monitor
- Clear phase progression
- Easy debugging

## 🔒 Additional Safeguards

### Concurrency Control
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```
- Cancels old runs when new push happens
- Prevents multiple version bumps for same branch

### Advisory Checks
```yaml
complexity:
  continue-on-error: true  # Won't block pipeline
```
- Complexity check is informational
- Won't prevent releases for minor complexity issues

### Conditional Execution
All deployment/release jobs check:
```yaml
if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```
- Only run on actual main branch pushes
- Skip for PRs, forks, manual triggers

## 📝 Migration Steps

### Step 1: Create New Workflow
- ✅ Created `.github/workflows/main.yml`

### Step 2: Test on Branch (Recommended)
```bash
# Create test branch
git checkout -b test/unified-workflow

# Push to test
git push origin test/unified-workflow

# Create PR to see it work
```

### Step 3: Delete Old Workflows
```bash
git rm .github/workflows/ci.yml
git rm .github/workflows/commitizen.yml
git rm .github/workflows/deploy-docs.yml
```

### Step 4: Update Documentation
- Update CONTRIBUTING.md to reference new workflow
- Document the phase structure

## 🎯 Expected Behavior

### Successful Push to Main:
1. ⏱️ **0-5 min**: Quality gates (lint, type, test) run in parallel
2. ⏱️ **5-7 min**: Docs build (after quality gates pass)
3. ⏱️ **7-8 min**: Docs deploy (after build succeeds)
4. ⏱️ **8-9 min**: Version bump (creates tag)
5. ⏱️ **9-10 min**: GitHub release created
6. ⏱️ **10-12 min**: PyPI publish (release.yml triggered by tag)

**Total**: ~12 minutes from push to PyPI publish ✅

### Failed Quality Gate:
1. ⏱️ **0-5 min**: Tests fail ❌
2. 🛑 **STOP**: Pipeline halts immediately
3. ✅ No version bump
4. ✅ No deployment
5. ✅ Fix issues, push again

## 🚀 Next Steps

1. **Test the workflow** on a branch first
2. **Delete old workflows** once validated
3. **Update branch protection** rules
4. **Document** for contributors

---

**Ready to migrate?** The new `main.yml` file has been created! 🎉
