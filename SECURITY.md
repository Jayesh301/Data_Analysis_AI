# Security Guide

## API Key Protection

### ✅ What's Protected

- `.env` files are excluded from git commits
- `.streamlit/secrets.toml` files are excluded from git commits
- Template files use placeholder values

### 🔒 Before Pushing to Git

1. **Check your .env file:**

   ```bash
   # Make sure your .env file contains placeholder text, not real API key
   cat .env
   # Should show: GEMINI_API_KEY=your_actual_api_key_here
   ```

2. **Check your secrets file:**

   ```bash
   # If you have a .streamlit/secrets.toml file, check it too
   cat .streamlit/secrets.toml
   # Should show placeholder, not real API key
   ```

3. **Verify .gitignore is working:**
   ```bash
   git status
   # Should NOT show .env or .streamlit/secrets.toml files
   ```

### 🚨 If API Key Was Exposed

If you accidentally committed your API key:

1. **Immediately rotate your API key** in the Google AI Studio
2. **Remove the commit from history:**
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch .env .streamlit/secrets.toml' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push to remove from remote:**
   ```bash
   git push origin --force
   ```

### 📝 Best Practices

- ✅ Use environment variables for local development
- ✅ Use Streamlit secrets for deployment
- ✅ Never hardcode API keys in source code
- ✅ Use placeholder values in templates
- ✅ Regularly rotate API keys
- ❌ Never commit real API keys to git
- ❌ Never share API keys in public repositories
