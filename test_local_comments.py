import os
import sys
import json
from github import Github

def main():
    # Get GitHub token from environment or prompt
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        github_token = input("Enter your GitHub token: ")
        if not github_token:
            print("ERROR: GitHub token is required")
            sys.exit(1)
    
    # Get repository and PR details
    repo_name = input("Enter repository name (owner/repo): ") or "Tomas-Jankauskas/ai-claude-code-reviewer"
    pr_number = int(input("Enter PR number: ") or "1")
    
    print(f"\nTesting GitHub API with repo: {repo_name}, PR: {pr_number}")
    
    try:
        # Initialize GitHub client
        gh = Github(github_token)
        
        # Get repository and PR
        print(f"Getting repository: {repo_name}")
        repo = gh.get_repo(repo_name)
        
        print(f"Getting PR #{pr_number}")
        pr = repo.get_pull(pr_number)
        print(f"PR title: {pr.title}")
        
        # Get PR files
        print("Getting PR files")
        files = list(pr.get_files())
        print(f"Found {len(files)} files in PR")
        
        for i, file in enumerate(files):
            print(f"\nFile {i+1}: {file.filename}")
            print(f"Status: {file.status}, Additions: {file.additions}, Deletions: {file.deletions}")
        
        # Choose method to test
        test_method = input("\nChoose test method (1: Issue comment, 2: PR review comment): ") or "1"
        
        if test_method == "1":
            # Test simple issue comment
            print("\nTesting issue comment (simplest approach)")
            comment = pr.create_issue_comment("Test comment from local script")
            print(f"Successfully created issue comment with ID: {comment.id}")
            
        elif test_method == "2":
            # Test review comment on a specific file
            print("\nTesting review comment")
            
            if not files:
                print("ERROR: No files found in PR")
                return
                
            file_choice = 0
            if len(files) > 1:
                file_choice = int(input(f"Choose file (1-{len(files)}): ")) - 1
                
            file = files[file_choice]
            print(f"Selected file: {file.filename}")
            
            # Get latest commit in PR
            commits = list(pr.get_commits())
            if not commits:
                print("ERROR: No commits found in PR")
                return
                
            latest_commit = commits[-1]
            print(f"Latest commit: {latest_commit.sha}")
            
            # Create a simple review with one comment
            body = "Test review comment"
            comments = [{
                "path": file.filename,
                "body": body,
                "line": file.additions  # Comment on the last line 
            }]
            
            print(f"Creating review with comment on {file.filename}")
            print(f"Comment details: {json.dumps(comments[0], indent=2)}")
            
            review = pr.create_review(
                commit=latest_commit,
                body="Test review",
                event="COMMENT",
                comments=comments
            )
            print(f"Successfully created review with ID: {review.id}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    main() 