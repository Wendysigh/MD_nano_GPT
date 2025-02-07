cd /path/to/your/project
git init
git add .
git status
git commit -m "update"

git config --global user.name "Wenqi"
git config --global user.email wzengad@connect.ust.hk
git commit --amend --reset-author

# Create a repository on GitHub
git remote add origin https://github.com/Wendysigh/MD_nano_GPT.git
git branch -M main
git push -u origin main


git rm -r --cached train/
git commit -m "Remove train/ directory from git tracking"
git push origin main