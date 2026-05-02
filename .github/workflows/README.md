# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the Hotel Booking Cancellation Predictor project.

## 📋 Available Workflows

### 1. CI/CD Pipeline (`ci.yml`)

**Triggers:**
- Push to `main`/`master` branch
- Pull requests to `main`/`master`/`dev`/`stage`

**Jobs:**
- **build-test:** Run tests across Python 3.11 and 3.12
  - Linting with Flake8
  - Code formatting check with Black
  - Unit and integration tests with coverage
  - Upload coverage reports to Codecov
  - Training pipeline smoke test
- **data-validation:** Validate data quality with Great Expectations

**Artifacts:**
- Coverage reports (HTML)
- Validation results

---

### 2. Deploy to HuggingFace (`deploy-huggingface.yml`)

**Triggers:**
- Push to `main`/`master` (when `Hotel-Booking/` changes)
- Manual workflow dispatch

**Jobs:**
- Export model for deployment
- Push to HuggingFace Spaces repository
- Automatic deployment to: https://huggingface.co/spaces/aelsayed1/Hotel-Booking

**Required Secret:**
- `HF_TOKEN` - HuggingFace API token with write access

**Setup:**
```bash
# Get HuggingFace token from: https://huggingface.co/settings/tokens
# Add to GitHub Secrets: Settings → Secrets → Actions → New repository secret
# Name: HF_TOKEN
# Value: hf_xxxxx...
```

---

### 3. Code Quality Checks (`code-quality.yml`)

**Triggers:**
- Pull requests to `main`/`master`/`dev`
- Push to `dev` branch

**Jobs:**
- **linting:** Flake8 linting with statistics
- **type-checking:** Check for print statements, TODOs, hardcoded credentials
- **security-scan:** 
  - Safety vulnerability scanning
  - TruffleHog secret detection

---

### 4. Model Training & Evaluation (`model-evaluation.yml`)

**Triggers:**
- Manual dispatch (with model selection)
- Weekly schedule (Sunday 2 AM UTC)
- Push to `main` (when model/feature code changes)

**Jobs:**
- **train-models:** Train multiple models
  - Default: baseline, logistic, xgboost, catboost
  - Generate evaluation reports and plots
- **benchmark-performance:** Test inference throughput

**Artifacts:**
- Trained models (30 days)
- Evaluation plots (30 days)
- Model report (90 days)

**Manual Trigger:**
```bash
# Go to Actions → Model Training & Evaluation → Run workflow
# Specify models: baseline,logistic,xgboost,catboost
```

---

### 5. Documentation & Reports (`documentation.yml`)

**Triggers:**
- Push to `main` (when docs/reports change)
- Manual workflow dispatch

**Jobs:**
- **generate-docs:** Create project statistics and model documentation
- **test-coverage-report:** Generate HTML coverage reports

**Artifacts:**
- Documentation files (90 days)
- Coverage HTML (30 days)
- Coverage summary (90 days)

---

### 6. Create Release (`release.yml`)

**Triggers:**
- Push tag matching `v*` (e.g., `v1.0.0`)
- Manual workflow dispatch

**Jobs:**
- Run full test suite
- Train final production model
- Generate release notes with metrics
- Create GitHub release with assets

**Release Assets:**
- `best_model.pkl` - Production model
- `evaluation_report.json` - Performance metrics
- `model_results.csv` - Model comparison
- Complete release package (tar.gz)

**Create Release:**
```bash
# Via Git tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or use GitHub Actions UI → Create Release → Run workflow
```

---

## 🔧 Setup Instructions

### 1. Required Secrets

Add these secrets in **Settings → Secrets → Actions**:

| Secret Name | Description | Required For |
|------------|-------------|--------------|
| `HF_TOKEN` | HuggingFace API token | HuggingFace deployment |
| `CODECOV_TOKEN` | Codecov upload token (optional) | Coverage reports |

### 2. Enable Workflows

1. Go to **Actions** tab in your repository
2. Enable GitHub Actions if not already enabled
3. Workflows will run automatically based on triggers

### 3. Manual Workflow Runs

Some workflows support manual triggering:

1. Go to **Actions** tab
2. Select the workflow
3. Click **Run workflow** button
4. Fill in any required inputs
5. Click **Run workflow**

---

## 📊 Monitoring Workflows

### View Workflow Runs
- Go to **Actions** tab
- Click on a workflow to see all runs
- Click on a specific run to see job details and logs

### Download Artifacts
- Open a workflow run
- Scroll to **Artifacts** section
- Click on artifact name to download

### Badges
Add status badges to your README:

```markdown
![CI/CD](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)
![Code Quality](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/code-quality.yml/badge.svg)
```

---

## 🚀 Best Practices

1. **Feature Development:**
   - Create feature branch from `dev`
   - Push commits → triggers Code Quality checks
   - Create PR to `dev` → triggers full CI/CD

2. **Model Updates:**
   - Update model code in feature branch
   - Manually run Model Evaluation workflow to test
   - Merge to `main` → auto-deploys to HuggingFace

3. **Releases:**
   - Merge to `main` and ensure CI passes
   - Create version tag (`v1.0.0`)
   - Release workflow auto-creates GitHub release

4. **Monitoring:**
   - Check Actions tab regularly
   - Review failed workflows immediately
   - Download artifacts for debugging

---

## 🔍 Troubleshooting

### Workflow Fails on Dependencies
```bash
# Update poetry.lock locally
poetry lock --no-update
git commit -am "Update poetry.lock"
git push
```

### HuggingFace Deployment Fails
- Verify `HF_TOKEN` secret is set correctly
- Check token has write permissions
- Ensure Space exists: https://huggingface.co/spaces/aelsayed1/Hotel-Booking

### Tests Timeout
- Check if test data is too large
- Increase timeout in workflow file
- Optimize slow tests

### Coverage Upload Fails
- Codecov token may be invalid
- Check if Codecov integration is enabled
- Verify coverage.xml is generated

---

## 📝 Workflow Modification

To modify workflows:

1. Edit `.github/workflows/<workflow>.yml`
2. Test changes in feature branch
3. Workflows run on push to validate syntax
4. Merge when working correctly

**Tip:** Use `act` to test workflows locally:
```bash
# Install act: https://github.com/nektos/act
act -l  # List workflows
act push  # Simulate push event
```

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)
- [Codecov Documentation](https://docs.codecov.com/)

---

**Last Updated:** May 2, 2026
