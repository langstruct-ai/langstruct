#!/bin/bash
set -e

# Helper script for releasing a new version
# Usage: ./scripts/release.sh [patch|minor|major]

BUMP_TYPE=${1:-patch}

echo "🚀 Starting release process..."
echo ""

# 1. Bump version
echo "📦 Bumping version ($BUMP_TYPE)..."
uv version --bump $BUMP_TYPE

# Get new version
VERSION=$(grep "^version" pyproject.toml | cut -d'"' -f2)
echo "✅ New version: v$VERSION"
echo ""

# 2. Update uv.lock with new version
echo "🔒 Updating uv.lock..."
uv sync --extra dev
echo ""

# 3. Commit version bump and lock file
echo "💾 Committing version bump..."
git add pyproject.toml uv.lock
git commit -m "Bump version to v$VERSION"
echo ""

# 4. Create and push tag
echo "🏷️  Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION"
echo ""

# 5. Push everything
echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin "v$VERSION"
echo ""

echo "✨ Done! Release v$VERSION is on its way."
echo ""
echo "GitHub Actions will now:"
echo "  - Run tests"
echo "  - Build the package"
echo "  - Publish to PyPI"
echo "  - Create a GitHub release"
echo ""
echo "Check progress at: https://github.com/langstruct-ai/langstruct/actions"
