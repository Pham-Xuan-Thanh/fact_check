#!/bin/bash
# Setup script for Fact-Checking Pipeline

echo "🚀 Setting up Fact-Checking Pipeline..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install system dependencies for Playwright/Chromium
echo "📦 Installing system dependencies..."
sudo apt-get install -qq -y \
    libgbm1 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libwayland-client0 \
    > /dev/null 2>&1

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install Playwright browsers
echo "📦 Installing Playwright browsers..."
playwright install chromium
playwright install-deps chromium

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Set your Google API key: export GOOGLE_API_KEY='your-api-key'"
echo "2. Run the server: python run.py"
echo ""
