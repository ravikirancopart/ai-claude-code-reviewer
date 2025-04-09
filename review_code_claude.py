import json
import os
import sys
from typing import List, Dict, Any
import anthropic
from github import Github
import difflib
import requests
import fnmatch
import re
from unidiff import Hunk, PatchedFile, PatchSet
from embeddings_store import GuidelinesStore
# from dotenv import load_dotenv // for local env only

# Load environment variables from .env file
# load_dotenv() // for local env only

# Set DEBUG to true to get more verbose logging
DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'

def debug_log(message):
    if DEBUG:
        print(f"DEBUG: {message}")

# Add detailed environment variable checks
try:
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    debug_log(f"GitHub token found with length: {len(GITHUB_TOKEN)}")
    debug_log(f"GitHub token first 4 chars: {GITHUB_TOKEN[:4]}...")
except KeyError:
    print("ERROR: GITHUB_TOKEN environment variable not set")
    sys.exit(1)

try:
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
    debug_log(f"Anthropic API key found with length: {len(ANTHROPIC_API_KEY)}")
    debug_log(f"Anthropic API key first 4 chars: {ANTHROPIC_API_KEY[:4]}...")
except KeyError:
    print("ERROR: ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

# Initialize GitHub and Anthropic clients
try:
    debug_log("Initializing GitHub client...")
    gh = Github(GITHUB_TOKEN)
    debug_log("GitHub client initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize GitHub client: {str(e)}")
    sys.exit(1)

try:
    debug_log("Initializing Anthropic client...")
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    debug_log("Anthropic client initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize Anthropic client: {str(e)}")
    sys.exit(1)

# Initialize guidelines store
try:
    debug_log("Initializing guidelines store...")
    guidelines_store = GuidelinesStore()
    guidelines_store.initialize_from_markdown('coding_guidelines.md')
    debug_log("Guidelines store initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize guidelines store: {str(e)}")
    sys.exit(1)

class PRDetails:
    def __init__(self, owner: str, repo: str, pull_number: int, title: str, description: str):
        self.owner = owner
        self.repo = repo
        self.pull_number = pull_number
        self.title = title
        self.description = description


def get_pr_details() -> PRDetails:
    """Retrieves details of the pull request from GitHub Actions event payload."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    
    # For local testing, use mock event file if GITHUB_EVENT_PATH is not set
    if event_path is None:
        event_path = ".github/test-data/pull_request_event.json"
        debug_log(f"Running locally, using mock event file: {event_path}")
    else:
        debug_log(f"Running in GitHub Actions, event path: {event_path}")
    
    try:
        with open(event_path, "r") as f:
            event_data = json.load(f)
        
        debug_log(f"Event data: {json.dumps(event_data, indent=2)}")

        # Handle PR labeled event
        if "pull_request" in event_data:
            pull_number = event_data["pull_request"]["number"]
            repo_full_name = event_data["repository"]["full_name"]
        else:
            print("ERROR: Unsupported event type, requires pull_request data")
            raise ValueError("Unsupported event type")

        debug_log(f"PR number: {pull_number}")
        debug_log(f"Repo: {repo_full_name}")

        owner, repo = repo_full_name.split("/")
        
        try:
            debug_log(f"Getting repo object for {repo_full_name}...")
            repo_obj = gh.get_repo(repo_full_name)
            debug_log(f"Getting PR #{pull_number}...")
            pr = repo_obj.get_pull(pull_number)
            debug_log(f"Successfully retrieved PR: {pr.title}")
            
            return PRDetails(owner, repo, pull_number, pr.title, pr.body)
        except Exception as e:
            print(f"ERROR: Failed to get PR details from GitHub: {str(e)}")
            raise
            
    except Exception as e:
        print(f"ERROR: Failed to parse event data: {str(e)}")
        raise


def get_diff(owner: str, repo: str, pull_number: int) -> str:
    """Fetches the diff of the pull request from GitHub API."""
    # Use the correct repository name format
    repo_name = f"{owner}/{repo}"
    print(f"Attempting to get diff for: {repo_name} PR#{pull_number}")

    try:
        debug_log(f"Getting repo object for {repo_name}...")
        repo = gh.get_repo(repo_name)
        debug_log(f"Getting PR #{pull_number}...")
        pr = repo.get_pull(pull_number)
        debug_log(f"Successfully retrieved PR: {pr.title}")

        # Use the GitHub API URL directly
        api_url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_number}"
        debug_log(f"Making API request to: {api_url}.diff")

        headers = {
            'Authorization': f'Bearer {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3.diff'
        }

        response = requests.get(f"{api_url}.diff", headers=headers)
        debug_log(f"API Response status code: {response.status_code}")

        if response.status_code == 200:
            diff = response.text
            print(f"Retrieved diff length: {len(diff) if diff else 0}")
            return diff
        else:
            print(f"ERROR: Failed to get diff. Status code: {response.status_code}")
            print(f"Response content: {response.text}")
            print(f"URL attempted: {api_url}.diff")
            print(f"Headers: {headers}")
            raise Exception(f"Failed to get diff from GitHub: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Exception getting diff: {str(e)}")
        raise

class Chunk:
    """Represents a chunk/hunk in a diff."""
    def __init__(self):
        self.content = ""
        self.changes = []
        self.source_start = 0
        self.source_length = 0
        self.target_start = 0
        self.target_length = 0

class Change:
    """Represents a single change line in a diff."""
    def __init__(self, content="", line_number=None):
        self.content = content
        self.line_number = line_number  # Line number in the target file
        self.diff_position = None  # Position within the diff file (for GitHub PR review API)

class File:
    """Represents a file in a diff."""
    def __init__(self):
        self.from_file = None
        self.to_file = None
        self.chunks = []

def parse_diff(diff_text: str) -> List[File]:
    """Parse a diff string into structured data."""
    files = []
    current_file = None
    current_chunk = None
    target_line_number = 0
    
    debug_log("Starting to parse diff...")
    try:
        for line in diff_text.splitlines():
            # Starting a new file
            if line.startswith("diff --git"):
                if current_file:
                    files.append(current_file)
                current_file = File()
                debug_log(f"New file found: {line}")
                continue
                
            if not current_file:
                continue
                
            # Get file names
            if line.startswith("--- "):
                current_file.from_file = line[4:].strip()
                debug_log(f"From file: {current_file.from_file}")
                continue
                
            if line.startswith("+++ "):
                current_file.to_file = line[4:].strip()
                debug_log(f"To file: {current_file.to_file}")
                continue
                
            # Start of a chunk/hunk
            if line.startswith("@@"):
                if current_chunk:
                    current_file.chunks.append(current_chunk)
                    
                current_chunk = Chunk()
                current_chunk.content = line
                
                # Parse the hunk header to get target line numbers
                # Format: @@ -a,b +c,d @@
                match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    source_start = int(match.group(1))
                    target_start = int(match.group(2))
                    current_chunk.source_start = source_start
                    current_chunk.target_start = target_start
                    target_line_number = target_start
                    debug_log(f"New chunk starting at line {target_line_number} (source line {source_start}): {line}")
                else:
                    target_line_number = 1  # Default if we can't parse
                    current_chunk.target_start = 1
                    current_chunk.source_start = 1
                    debug_log(f"Warning: Could not parse line numbers from hunk header: {line}")
                    
                continue
                
            if current_chunk:
                current_chunk.content += "\n" + line
                
                # Store the change with its position in the file and in the diff
                if line.startswith(" ") or line.startswith("+"):
                    # Both context and added lines increment the target file line number
                    change = Change(content=line, line_number=target_line_number)
                    current_chunk.changes.append(change)
                    target_line_number += 1
                elif line.startswith("-"):
                    # Removed lines don't affect target file line numbers
                    change = Change(content=line)
                    current_chunk.changes.append(change)
        
        # Add the last file and chunk if any
        if current_file:
            if current_chunk:
                current_file.chunks.append(current_chunk)
            files.append(current_file)
        
        # Calculate positions for GitHub's PR review API
        # GitHub needs position to be relative to the start of the diff
        for file in files:
            position_counter = 0
            for chunk in file.chunks:
                # The hunk header line counts as position 1
                position_counter += 1
                
                for i, change in enumerate(chunk.changes):
                    # Each change's position is its index in the entire diff file
                    position_counter += 1
                    # Store the position for later use
                    change.diff_position = position_counter
                    
                    # Debug information for the first few and last few changes
                    if i < 3 or i >= len(chunk.changes) - 3:
                        line_str = f"line {change.line_number}" if change.line_number else "removed line"
                        debug_log(f"Change {i} at position {position_counter} ({line_str}): {change.content[:30]}")
        
        debug_log(f"Diff parsing complete. Found {len(files)} files.")
        for file in files:
            debug_log(f"File: {file.to_file} with {len(file.chunks)} chunks")
            
        return files
    except Exception as e:
        print(f"ERROR: Failed to parse diff: {str(e)}")
        raise

def create_prompt(file: File, chunk: Chunk, pr_details: PRDetails) -> str:
    """Creates a prompt for Claude to review the code."""
    try:
        # Get relevant guidelines based on the code being reviewed
        debug_log(f"Getting relevant guidelines for {file.to_file}...")
        relevant_guidelines = guidelines_store.get_relevant_guidelines(
            code_snippet=chunk.content,
            file_path=file.to_file
        )
        
        guidelines_text = "\n".join(relevant_guidelines)
        debug_log(f"Found {len(relevant_guidelines)} relevant guidelines")
        
        # Map the changes to a format that includes line numbers for easier review
        formatted_changes = []
        for change in chunk.changes:
            line_prefix = f"{change.line_number} " if change.line_number else ""
            formatted_changes.append(f"{line_prefix}{change.content}")
        
        formatted_chunk = "\n".join(formatted_changes)
        debug_log(f"Formatted diff with {len(chunk.changes)} changes")
        
        prompt = f"""Your task is reviewing pull requests according to our coding guidelines. Instructions:
        - Provide the response in following JSON format: {{"reviews": [{{"lineNumber": <line_number>, "reviewComment": "<review comment>"}}]}}
        - The lineNumber should reference the line numbers shown at the beginning of each line in the diff
        - Only comment on the lines that start with '+' in the diff (added lines)
        - Only provide comments if there is something to improve
        - Use GitHub Markdown in comments
        - Focus on bugs, security issues, performance problems, and adherence to our coding guidelines
        - IMPORTANT: NEVER suggest adding comments to the code

Here are the relevant coding guidelines for this code:

{guidelines_text}

Review the following code diff in the file "{file.to_file}" and take the pull request title and description into account when writing the response.

Pull request title: {pr_details.title}
Pull request description:

---
{pr_details.description or 'No description provided'}
---

Git diff to review (format: line_number content):

```diff
{formatted_chunk}
```
"""
        debug_log(f"Created prompt of length {len(prompt)}")
        return prompt
    except Exception as e:
        print(f"ERROR: Failed to create prompt: {str(e)}")
        raise

def get_ai_response(prompt: str) -> List[Dict[str, str]]:
    """Sends the prompt to Claude API and retrieves the response."""
    model = os.environ.get('CLAUDE_MODEL', 'claude-3-5-sonnet-20240620')
    debug_log(f"Using Claude model: {model}")

    print("===== The prompt sent to Claude is: =====")
    print(prompt)
    try:
        debug_log("Sending prompt to Claude API...")
        response = claude_client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.7,
            system="You are an expert code reviewer. Provide feedback in the requested JSON format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        debug_log("Received response from Claude API")

        response_text = response.content[0].text
        debug_log(f"Raw response text length: {len(response_text)}")
        
        # Extract JSON if it's wrapped in code blocks
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()

        print(f"Cleaned response text: {response_text}")

        try:
            data = json.loads(response_text)
            debug_log(f"Successfully parsed JSON from response")

            if "reviews" in data and isinstance(data["reviews"], list):
                reviews = data["reviews"]
                debug_log(f"Found {len(reviews)} reviews in response")
                
                valid_reviews = []
                for review in reviews:
                    if "lineNumber" in review and "reviewComment" in review:
                        valid_reviews.append(review)
                        debug_log(f"Valid review for line {review['lineNumber']}: {review['reviewComment'][:50]}...")
                    else:
                        print(f"Invalid review format: {review}")
                return valid_reviews
            else:
                print("ERROR: Response doesn't contain valid 'reviews' array")
                print(f"Response content: {data}")
                return []
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON response: {e}")
            print(f"Raw response: {response_text}")
            return []
    except Exception as e:
        print(f"ERROR: Failed during Claude API call: {str(e)}")
        raise

def create_comment(file: File, chunk: Chunk, ai_responses: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Creates comment objects from AI responses."""
    comments = []
    
    debug_log(f"Creating comments for {len(ai_responses)} AI responses")
    
    # Create a lookup of line numbers to changes
    line_map = {}
    for change in chunk.changes:
        if change.line_number is not None:
            line_map[change.line_number] = change
    
    # Debug info for line map
    debug_log("\nLine map:")
    for line_num, change in sorted(line_map.items()):
        debug_log(f"Line {line_num}: {change.content}")
    
    for ai_response in ai_responses:
        try:
            line_number = int(ai_response["lineNumber"])
            debug_log(f"Processing AI response for line {line_number}")
            
            # Check if the line is in our map
            if line_number not in line_map:
                debug_log(f"Warning: Line {line_number} not found in the diff")
                continue
                
            # Get the change for this line
            change = line_map[line_number]
            
            debug_log(f"@@@@TEST change: {change}")
            
            # Ensure the line is an added line (starts with +)
            if not change.content.startswith("+"):
                debug_log(f"Warning: Line {line_number} is not an added line: {change.content}")
                continue
            
            # GitHub expects paths without 'a/' or 'b/' prefixes
            path = file.to_file
            debug_log(f"@@@@TEST path: {path}")
            if path.startswith('a/'):
                path = path[2:]
                debug_log(f"Removed a/ prefix from path: {path}")
            elif path.startswith('b/'):
                path = path[2:]
                debug_log(f"Removed b/ prefix from path: {path}")
            
            comment = {
                "body": ai_response["reviewComment"],
                "path": path,
                "line": line_number,  # This is more reliable than position
            }
            
            debug_log(f"@@@@TEST comment: {comment}")
            debug_log(f"Created comment for {path} at line {line_number}")
            comments.append(comment)
            debug_log(f'@@@@TEST comments: {comments}')

        except (KeyError, TypeError, ValueError) as e:
            debug_log(f"Error creating comment from AI response: {e}, Response: {ai_response}")
    
    debug_log(f"Created {len(comments)} valid comments")
    return comments

def analyze_code(parsed_diff: List[File], pr_details: PRDetails) -> List[Dict[str, Any]]:
    """Analyzes the code changes using Claude and generates review comments."""
    print("Starting analyze_code...")
    print(f"Number of files to analyze: {len(parsed_diff)}")
    comments = []

    for file in parsed_diff:
        print(f"\nProcessing file: {file.to_file}")
        
        if not file.to_file or file.to_file == "/dev/null":
            debug_log(f"Skipping file with invalid path: {file.to_file}")
            continue
            
        # Process each chunk in the file
        for chunk_index, chunk in enumerate(file.chunks):
            debug_log(f"Processing chunk {chunk_index+1}/{len(file.chunks)}")
            prompt = create_prompt(file, chunk, pr_details)
            print(f"Created prompt of length {len(prompt)}")
            
            # Get AI response
            ai_responses = get_ai_response(prompt)
            print(f"AI generated {len(ai_responses)} review comments")
            
            # Create comments from AI responses
            new_comments = create_comment(file, chunk, ai_responses)
            comments.extend(new_comments)
            print(f"Added {len(new_comments)} new comments")

    print(f"\nFinal comments list: {len(comments)} comments")
    if len(comments) > 0:
        debug_log("Sample of comments:")
        for i, comment in enumerate(comments[:3]):  # Show first 3 comments
            debug_log(f"Comment {i+1}: {comment['path']}:{comment.get('line', 'N/A')} - {comment['body'][:50]}...")
    
    return comments

def create_review_comment(
    owner: str,
    repo: str,
    pull_number: int,
    comments: List[Dict[str, Any]],
):
    """Creates a pull request review with comments on specific lines."""
    print(f"Creating PR review with {len(comments)} comments")
    
    if not comments:
        print("WARNING: No comments to post, skipping")
        return
        
    try:
        # Initialize GitHub client and get repository
        debug_log(f"Getting repo object for {owner}/{repo}...")
        repo_obj = gh.get_repo(f"{owner}/{repo}")
        debug_log(f"Getting PR #{pull_number}...")
        pr = repo_obj.get_pull(pull_number)
        debug_log(f"Successfully retrieved PR: {pr.title}")
        
        # Format comments for the review
        formatted_comments = []
        
        for comment in comments:
            path = comment.get('path')
            line = comment.get('line')
            body = comment.get('body')
            
            if not path or not line or not body:
                debug_log(f"Skipping comment with missing data: path={path}, line={line}")
                continue
                
            formatted_comment = {
                'path': path,
                'line': line,
                'body': body
            }
            
            debug_log(f"Adding comment for {path}:{line}")
            formatted_comments.append(formatted_comment)
        
        if not formatted_comments:
            print("WARNING: No valid comments to post")
            return
            
        # Create the pull request review with all comments
        review = pr.create_review(
            body="Code review by Claude",
            event="COMMENT",
            comments=formatted_comments
        )
        
        print(f"Successfully created PR review with ID: {review.id}")
        return review.id
        
    except Exception as e:
        print(f"ERROR: Failed to create PR review: {str(e)}")
        
        # Try fallback method - post comments individually
        try:
            debug_log("Using fallback: Posting comments individually")
            comment_ids = []
            
            for comment in comments:
                path = comment.get('path')
                line = comment.get('line')
                body = comment.get('body')
                
                if not path or not line or not body:
                    continue
                    
                try:
                    pr_comment = pr.create_comment(
                        body=body,
                        path=path,
                        line=line
                    )
                    comment_ids.append(pr_comment.id)
                    debug_log(f"Created individual comment on {path}:{line}")
                except Exception as comment_error:
                    debug_log(f"Failed to create individual comment: {str(comment_error)}")
            
            if comment_ids:
                print(f"Created {len(comment_ids)} individual comments")
                return comment_ids
            else:
                raise Exception("Failed to create any individual comments")
                
        except Exception as e2:
            debug_log(f"Individual comment fallback failed: {str(e2)}")
            
            # Final fallback - post a consolidated issue comment
            try:
                debug_log("Using final fallback: Posting consolidated issue comment")
                
                # Group comments by file
                comments_by_file = {}
                for comment in comments:
                    file_path = comment.get('path', 'Unknown')
                    if file_path not in comments_by_file:
                        comments_by_file[file_path] = []
                    comments_by_file[file_path].append(comment)
                
                # Generate the comment body
                comment_body = "# Claude Code Review Results\n\n"
                
                for file_path, file_comments in comments_by_file.items():
                    comment_body += f"## File: {file_path}\n\n"
                    
                    # Sort comments by line number
                    file_comments.sort(key=lambda c: c.get('line', 0))
                    
                    for comment in file_comments:
                        line_num = comment.get('line', 'N/A')
                        comment_body += f"### Line {line_num}\n\n"
                        comment_body += f"{comment.get('body', '')}\n\n"
                        comment_body += "---\n\n"
                
                # Create the fallback comment
                fallback_comment = pr.create_issue_comment(comment_body)
                print(f"Created fallback consolidated comment with ID: {fallback_comment.id}")
                return [fallback_comment.id]
                
            except Exception as e3:
                debug_log(f"All fallback methods failed: {str(e3)}")
                raise Exception(f"Failed to create any type of comments: {str(e)}")

def post_comments_to_pr(
    owner: str,
    repo: str,
    pull_number: int,
    comments: List[Dict[str, Any]],
):
    """
    DEPRECATED: This function has been replaced by create_review_comment.
    Forwards to the new function for backward compatibility.
    """
    print("WARNING: Using deprecated post_comments_to_pr function")
    print("Please update code to use create_review_comment instead")
    return create_review_comment(owner, repo, pull_number, comments)

def create_review_comment_deprecated(
    owner: str,
    repo: str,
    pull_number: int,
    comments: List[Dict[str, Any]],
):
    """
    DEPRECATED: This function has been replaced by create_review_comment.
    Forwards to the new function for backward compatibility.
    """
    print("WARNING: Using deprecated create_review_comment_deprecated function")
    print("Please update code to use create_review_comment instead")
    return create_review_comment(owner, repo, pull_number, comments)

def main():
    """Main function to execute the code review process."""
    try:
        print("=== Starting Claude Code Reviewer ===")
        debug_log("Starting code review process with DEBUG enabled")
        
        # Print current environment for debugging
        debug_log(f"Python version: {sys.version}")
        debug_log(f"Current directory: {os.getcwd()}")
        debug_log(f"Environment variables: {[k for k in os.environ.keys() if not k.startswith('_')]}")
        
        pr_details = get_pr_details()
        debug_log(f"Got PR details: {pr_details.__dict__}")

        diff = get_diff(pr_details.owner, pr_details.repo, pr_details.pull_number)
        
        debug_log(f"Got diff of length: {len(diff)}")
        
        if not diff:
            print("WARNING: No diff found, nothing to review")
            return

        parsed_diff = parse_diff(diff)
        debug_log(f"Parsed diff into {len(parsed_diff)} files")

        # Get and clean exclude patterns
        exclude_patterns_raw = os.environ.get("INPUT_EXCLUDE", "")
        debug_log(f"Raw exclude patterns: {exclude_patterns_raw}")
        
        exclude_patterns = []
        if exclude_patterns_raw and exclude_patterns_raw.strip():
            exclude_patterns = [p.strip() for p in exclude_patterns_raw.split(",") if p.strip()]
        debug_log(f"Processed exclude patterns: {exclude_patterns}")

        # Filter files
        filtered_diff = []
        for file in parsed_diff:
            file_path = file.to_file
            should_exclude = any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns)
            if should_exclude:
                debug_log(f"Excluding file: {file_path}")
                continue
            filtered_diff.append(file)
            debug_log(f"Including file: {file_path}")

        debug_log(f"Files to analyze after filtering: {[f.to_file for f in filtered_diff]}")
        
        if not filtered_diff:
            print("WARNING: No files to analyze after filtering")
            return
            
        comments = analyze_code(filtered_diff, pr_details)
        debug_log(f"Generated {len(comments)} comments")
        
        if comments:
            try:
                # Use the create_review_comment function to create a PR review
                review_id = create_review_comment(
                    pr_details.owner, pr_details.repo, pr_details.pull_number, comments
                )
                print(f"Successfully posted review with ID: {review_id}")
            except Exception as e:
                print(f"ERROR: Failed to post comments: {str(e)}")
                sys.exit(1)  # Exit with error code
        else:
            print("No issues found, no comments to post")
    except Exception as error:
        print(f"ERROR in main: {str(error)}")
        sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL ERROR: {str(error)}")
        sys.exit(1)  # Exit with error code 