# OpenAI Code Reviewer

# Claude Code Reviewer

A GitHub Action that automatically reviews pull requests using Anthropic's Claude 3.5 Sonnet.

## Features

- Review your PRs using Anthropic's Claude API
- Give use comments and suggestions to improve the source codes
- Automatically fetch and update coding guidelines from Confluence

## Setup

1. To use this GitHub Action, you need an Anthropic API key. If you don't have one, sign up for an API key
   at [Anthropic Console](https://console.anthropic.com/).

2. Add the Anthropic API key as a GitHub Secret in your repository with the name `ANTHROPIC_API_KEY`. You can find more
   information about GitHub Secrets [here](https://docs.github.com/en/actions/reference/encrypted-secrets).

3. (Optional) If you want to use auto-refreshing guidelines from Confluence, add the following secrets:
   - `JIRA_USERNAME`: Your Atlassian username (email)
   - `JIRA_API_TOKEN`: Your Atlassian API token
   - `CONFLUENCE_URL`: The URL to your Confluence page with coding guidelines

4. Create a `.github/workflows/code-review.yml` file in your repository and add the following content:

```yaml
name: Claude Code Reviewer

on:
  issue_comment:
    types: [created]

permissions: write-all

jobs:
  claude-code-review:
    runs-on: ubuntu-latest
    if: |
      github.event.issue.pull_request &&
      contains(github.event.comment.body, '/claude-review')
    steps:
      - name: PR Info
        run: |
          echo "Comment: ${{ github.event.comment.body }}"
          echo "Issue Number: ${{ github.event.issue.number }}"
          echo "Repository: ${{ github.repository }}"

      - name: Checkout Repo
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Get PR Details
        id: pr
        run: |
          PR_JSON=$(gh api repos/${{ github.repository }}/pulls/${{ github.event.issue.number }})
          echo "head_sha=$(echo $PR_JSON | jq -r .head.sha)" >> $GITHUB_OUTPUT
          echo "base_sha=$(echo $PR_JSON | jq -r .base.sha)" >> $GITHUB_OUTPUT
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - uses: Tomas-Jankauskas/ai-code-reviewer@main
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          CLAUDE_MODEL: claude-3-5-sonnet-20240620 # Optional, default is `claude-3-5-sonnet-20240620`
          EXCLUDE: "*.md,*.txt,package-lock.json,*.yml,*.yaml"
          # Optional Confluence integration for guidelines
          JIRA_USERNAME: ${{ secrets.JIRA_USERNAME }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
          CONFLUENCE_URL: ${{ secrets.CONFLUENCE_URL }}
```
> If you don't set `CLAUDE_MODEL`, the default model is `claude-3-5-sonnet-20240620`. You can also use other Claude models like `claude-3-opus-20240229` for more advanced capabilities.

5. Commit codes to your repository, and working on your pull requests.
6. When you're ready to review the PR, you can trigger the workflow by commenting `/claude-review` in the PR.

## How It Works

This GitHub Action uses the Anthropic Claude API to provide code review feedback. It works by:

1. **Analyzing the changes**: It grabs the code modifications from your pull request and filters out any files you don't want reviewed.
2. **Consulting the Claude model**: It sends chunks of the modified code to Claude for analysis.
3. **Providing feedback**: Claude examines the code and generates review comments.
4. **Delivering the review**: The Action adds the comments directly to your pull request on GitHub.

## Auto-Refreshing Guidelines

The action can automatically fetch and use your team's coding guidelines from Confluence:

1. **Weekly Updates**: Every 7 days, the action will check for updates to your guidelines.
2. **Confluence Integration**: Guidelines are fetched directly from your specified Confluence page.
3. **RAG-based Relevance**: The action uses Retrieval-Augmented Generation to apply only the most relevant guidelines to each code section.
4. **Fallback**: If Confluence is unavailable, the action will use the most recent local copy of guidelines.

### Configuring Guidelines

To use your own guidelines:

1. Create a Confluence page with your coding standards.
2. Set the `CONFLUENCE_URL`, `JIRA_USERNAME`, and `JIRA_API_TOKEN` secrets in your repository.
3. The action will automatically fetch and format the guidelines.

You can also manually provide a `coding_guidelines.md` file in your repository, which will be used if the Confluence integration is not configured.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.