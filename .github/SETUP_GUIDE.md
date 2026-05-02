# GitHub Actions Setup Guide

## ✅ What's Been Implemented

Your repository now has **6 automated workflows** that handle testing, deployment, code quality, and releases.

### 1. 🔄 CI/CD Pipeline
- **File:** `.github/workflows/ci.yml`
- **Runs on:** Every push and pull request
- **What it does:**
  - Tests on Python 3.11 & 3.12
  - Runs linting and formatting checks
  - Generates code coverage reports
  - Validates data quality
  - Uploads coverage to Codecov

### 2. 🚀 HuggingFace Deployment
- **File:** `.github/workflows/deploy-huggingface.yml`
- **Runs on:** Push to main (when Hotel-Booking/ changes)
- **What it does:**
  - Exports model for deployment
  - Automatically pushes to HuggingFace Space
  - Deploys to: https://huggingface.co/spaces/aelsayed1/Hotel-Booking

### 3. 🔍 Code Quality Checks
- **File:** `.github/workflows/code-quality.yml`
- **Runs on:** Pull requests
- **What it does:**
  - Flake8 linting
  - Black formatting check
  - Security vulnerability scanning
  - Secret detection

### 4. 🤖 Model Training & Evaluation
- **File:** `.github/workflows/model-evaluation.yml`
- **Runs on:** Weekly schedule + manual trigger
- **What it does:**
  - Trains multiple models
  - Generates evaluation reports
  - Benchmarks inference performance
  - Creates model artifacts

### 5. 📚 Documentation Generation
- **File:** `.github/workflows/documentation.yml`
- **Runs on:** Push to main (docs changes)
- **What it does:**
  - Generates project statistics
  - Creates model performance docs
  - Generates coverage reports

### 6. 📦 Release Creation
- **File:** `.github/workflows/release.yml`
- **Runs on:** Version tags (v*.*.*)
- **What it does:**
  - Trains production model
  - Creates GitHub release
  - Packages model artifacts
  - Generates release notes

---

## 🔧 Required Setup

### Step 1: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click **Settings** → **Actions** → **General**
3. Under "Actions permissions", select **Allow all actions**
4. Under "Workflow permissions", select **Read and write permissions**
5. Click **Save**

### Step 2: Add HuggingFace Token

For automated deployment to work, add your HuggingFace token:

1. Get your token from: https://huggingface.co/settings/tokens
   - Click **New token**
   - Name: `github-actions`
   - Type: **Write**
   - Copy the token

2. In your GitHub repo: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `HF_TOKEN`
5. Value: Paste your HuggingFace token
6. Click **Add secret**

### Step 3: (Optional) Add Codecov Token

For coverage reports:

1. Go to: https://codecov.io/
2. Sign in with GitHub
3. Add your repository
4. Copy the upload token
5. Add as GitHub secret: `CODECOV_TOKEN`

---

## 🎯 How to Use

### Running Tests Automatically
- **Push code** → CI/CD runs automatically
- **Create PR** → All checks run
- Check **Actions** tab to see results

### Deploying to HuggingFace
**Option 1: Automatic (Recommended)**
```bash
# Make changes to Hotel-Booking/
git add Hotel-Booking/
git commit -m "Update app"
git push

# Workflow automatically deploys to HuggingFace
```

**Option 2: Manual**
1. Go to **Actions** → **Deploy to HuggingFace**
2. Click **Run workflow**
3. Select branch
4. Click **Run workflow**

### Training Models
**Manual Training:**
1. Go to **Actions** → **Model Training & Evaluation**
2. Click **Run workflow**
3. Enter models: `baseline,logistic,xgboost,catboost`
4. Click **Run workflow**

**Scheduled Training:**
- Runs automatically every Sunday at 2 AM UTC
- Results saved as artifacts

### Creating a Release
```bash
# Tag your release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Release workflow runs automatically
# Creates GitHub release with model artifacts
```

---

## 📊 Viewing Results

### Check Workflow Status
1. Go to **Actions** tab
2. Click on a workflow run
3. View job details and logs

### Download Artifacts
1. Open a completed workflow run
2. Scroll to **Artifacts** section
3. Click to download:
   - Coverage reports
   - Trained models
   - Evaluation plots
   - Documentation

### Add Status Badges

Add to your `README.md`:

```markdown
![CI/CD](https://github.com/omarhashem80/HotelBookingCancellationPredictor/actions/workflows/ci.yml/badge.svg)
![Code Quality](https://github.com/omarhashem80/HotelBookingCancellationPredictor/actions/workflows/code-quality.yml/badge.svg)
![Deploy](https://github.com/omarhashem80/HotelBookingCancellationPredictor/actions/workflows/deploy-huggingface.yml/badge.svg)
```

---

## 🐛 Troubleshooting

### Workflow Fails: "HF_TOKEN not found"
- Verify secret is added: Settings → Secrets → Actions
- Name must be exactly: `HF_TOKEN`
- Token must have write permissions

### Tests Fail Locally but Pass in CI
```bash
# Update dependencies to match CI
poetry install
poetry lock --no-update
```

### HuggingFace Deployment Fails
1. Check Space exists: https://huggingface.co/spaces/aelsayed1/Hotel-Booking
2. Verify token has write access
3. Check workflow logs for specific error

### Coverage Upload Fails
- Codecov token may be invalid
- Check if token is added correctly
- Verify Codecov GitHub app is installed

---

## 🎉 What's Next?

Your workflows are now active! They will:

- ✅ Run tests on every push
- ✅ Check code quality on PRs
- ✅ Deploy to HuggingFace automatically
- ✅ Train models weekly
- ✅ Create releases with artifacts

### Recommended Workflow

1. **Feature Development:**
   ```bash
   git checkout -b feature/my-feature
   # Make changes
   git push origin feature/my-feature
   # Create PR → CI runs automatically
   ```

2. **Model Updates:**
   - Update model code
   - Run **Model Training** workflow manually
   - Review results in artifacts
   - Merge when satisfied

3. **Production Release:**
   ```bash
   git checkout master
   git pull
   git tag -a v1.0.0 -m "Production release"
   git push origin v1.0.0
   # Release created automatically
   ```

---

## 📝 Customization

### Change Schedule
Edit `.github/workflows/model-evaluation.yml`:
```yaml
schedule:
  - cron: '0 2 * * 0'  # Change time/day
```

### Add More Models
Edit workflow inputs:
```yaml
default: 'baseline,logistic,xgboost,catboost,histboost'
```

### Modify Test Coverage
Edit `.github/workflows/ci.yml`:
```yaml
run: poetry run pytest --cov=src --cov-fail-under=80
```

---

## 📚 Resources

- [Workflow README](.github/workflows/README.md) - Detailed docs
- [GitHub Actions Docs](https://docs.github.com/actions)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)

---

**Setup Complete!** 🎊

Check the **Actions** tab to see your workflows running.
