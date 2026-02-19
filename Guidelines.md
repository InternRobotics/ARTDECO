# 🛠️ Collaboration Guidelines

These guidelines define how we work together in this repository. Please read and follow them to ensure smooth and efficient collaboration.

---

## 🔀 Merge Guidelines

- ✅ **Describe Your Changes Clearly**  
  Every merge should include a summary of what was changed and why. Avoid vague descriptions.

- 🚫 **Do Not Upload Temporary or Debug Scripts**  
  Avoid committing one-off scripts, debug prints, or experimental files unless they are clearly marked and essential to a shared workflow.

- 📝 **Use Descriptive Commit Messages**  
  Commit messages should clearly describe the changes introduced. Prefer messages like `fix: resolve edge case in depth handler` over `update` or `wip`.

- ⚠️ **Avoid Esoteric or Cryptic Code Comments**  
  Comments should be understandable to all contributors. Use clear, plain language—no unexplained shorthand or inside jokes.

- 🌐 **Prefer English for All Messages**  
  For consistency and collaboration across different contributors, use English in commit messages, comments, and PRs whenever possible.

---

## 🚧 Development Workflow

- 🌿 **Work in Feature Branches**  
  Each contributor should create and work in their own feature, topic, or named branch. Do not commit directly to `main`.

- 🔁 **Preserve Existing Functionality**  
  When adding new features, make sure not to break existing ones.  
  **Best practice:** Add a flag, toggle, or CLI argument to enable new behaviors.

- 📌 **Milestone-Based Pull Requests**  
  Open a Pull Request when you've reached a meaningful milestone. Use the PR to explain your changes and provide context.

- 🧑‍⚖️ **All Merges to `main` Must Be Reviewed**  
  You must request and receive approval from **at least one other contributor** before merging into `main`.

- 📖 **Commit Style (Optional, Recommended)**  
  Though we are not doing automated changelog,consider using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to improve readability.
